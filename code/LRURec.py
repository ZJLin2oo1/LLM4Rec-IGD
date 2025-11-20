import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast

# Paths (update to your actual files)
TRAIN_CSV_PATH = './code/train/Books_5_2017-10-2018-11.csv'
VALID_CSV_PATH = './code/valid/Books_5_2017-10-2018-11.csv'
TEST_CSV_PATH = './code/test/Books_5_2017-10-2018-11.csv'

class RecommendationDataset(Dataset):
    def __init__(self, csv_path, max_len, item_vocab_size):
        """
        Args:
            csv_path (str): Path to train.csv, valid.csv, or test.csv.
            max_len (int): Max sequence length L.
            item_vocab_size (int): Total number of items V (padding ID uses 0).
        """
        self.df = pd.read_csv(csv_path)
        # Safely convert stringified lists to real lists
        self.df['history_item_id'] = self.df['history_item_id'].apply(ast.literal_eval)

        self.max_len = max_len
        # Use 0 as padding_id
        self.padding_id = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        history_ids = row['history_item_id']
        target_id = row['item_id']

        # --- Build input and labels ---
        # Input: left-pad/truncate to max_len
        input_seq = history_ids
        padded_input = [self.padding_id] * (self.max_len - len(input_seq)) + input_seq
        padded_input = padded_input[-self.max_len:]

        # Labels: shifted input plus target at the end
        labels = padded_input[1:] + [target_id]

        return torch.LongTensor(padded_input), torch.LongTensor(labels)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = LRUEmbedding(self.args)
        self.model = LRUModel(self.args)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if 'layer_norm' in n or 'params_log' in n:
                    continue
                if torch.is_complex(p):
                    p.real.uniform_(2 * l - 1, 2 * u - 1)
                    p.imag.uniform_(2 * l - 1, 2 * u - 1)
                    p.real.erfinv_()
                    p.imag.erfinv_()
                    p.real.mul_(std * math.sqrt(2.))
                    p.imag.mul_(std * math.sqrt(2.))
                    p.real.add_(mean)
                    p.imag.add_(mean)
                else:
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def forward(self, x, labels=None):
        x, mask = self.embedding(x)
        return self.model(x, self.embedding.token.weight, mask, labels=labels)

class LRUEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 1
        embed_size = args.bert_hidden_units

        self.token = nn.Embedding(vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)

    def get_mask(self, x):
        return (x > 0)

    def forward(self, x):
        mask = self.get_mask(x)
        x = self.token(x)
        return self.layer_norm(self.embed_dropout(x)), mask

class LRUModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.bert_hidden_units
        layers = args.bert_num_blocks

        self.lru_blocks = nn.ModuleList([LRUBlock(self.args) for _ in range(layers)])
        self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 1))

    def forward(self, x, embedding_weight, mask, labels=None):
        # Left pad to power of 2 for the parallel algorithm
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks + position-wise FFN
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        x = x[:, -seq_len:]  # B x L x D

        # Prediction layer
        if self.args.dataset_code != 'xlong':
            scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
            return scores, None
        else:
            assert labels is not None
            if self.training:
                num_samples = self.args.negative_sample_size
                samples = torch.randint(1, self.args.num_items + 1, size=(*x.shape[:2], num_samples,))
                all_items = torch.cat([samples.to(labels.device), labels.unsqueeze(-1)], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b l i d -> b l i', x, sampled_embeddings) + self.bias[all_items]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_
            else:
                num_samples = self.args.xlong_negative_sample_size
                samples = torch.randint(1, self.args.num_items + 1, size=(x.shape[0], num_samples,))
                all_items = torch.cat([samples.to(labels.device), labels], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b i d -> b l i', x, sampled_embeddings) + self.bias[all_items.unsqueeze(1)]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_.reshape(labels.shape)

class LRUBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = args.bert_hidden_units
        self.lru_layer = LRULayer(d_model=hidden_size, dropout=args.bert_attn_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden_size, d_ff=hidden_size * 4, dropout=args.bert_dropout)

    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x

class LRULayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, use_bias=True, r_min=0.8, r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # Initialize nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C (projections)
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        self.out_vector = nn.Identity()

        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm (optimized for zero padding)
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)
        mask_ = mask.reshape(B * L // l, l)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]
        if i > 1:
            lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # Compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu

        # Parallel scan across sequence length
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)

        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import math
import ast
import os
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def evaluate(model, dataloader, device, ks=[5, 10]):
    """
    Evaluate model on given dataloader with HR@K and NDCG@K.
    """
    model.eval()
    hr_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}

    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs = seqs.to(device)
            logits, _ = model(seqs)  # (B, L, V)
            scores = logits[:, -1, :]  # (B, V)
            true_labels = labels[:, -1]  # (B)

            # Mask items that already appeared in the sequence
            scores.scatter_(1, seqs.to(device), -1e9)

            _, topk_indices = torch.topk(scores, k=max(ks), dim=-1)
            true_labels_expanded = true_labels.unsqueeze(-1).expand_as(topk_indices)
            hits = (topk_indices == true_labels_expanded.to(device))

            for k in ks:
                hits_k = hits[:, :k]
                hr_k = torch.any(hits_k, dim=1).float().mean().item()
                hr_scores[k].append(hr_k)

                if hits_k.sum() > 0:
                    hit_ranks = torch.where(hits_k)[1] + 1
                    dcg = (1.0 / torch.log2(hit_ranks.float() + 1)).sum()
                    ndcg_k = dcg / len(true_labels)
                    ndcg_scores[k].append(ndcg_k.item())
                else:
                    ndcg_scores[k].append(0.0)

    final_metrics = {}
    for k in ks:
        final_metrics[f'HR@{k}'] = np.mean(hr_scores[k])
        final_metrics[f'NDCG@{k}'] = np.mean(ndcg_scores[k])
    return final_metrics


# ---------------- Configuration ----------------
MAX_SEQUENCE_LENGTH = 50

args = argparse.Namespace()
args.max_len = 50
args.num_items = 41723
args.bert_hidden_units = 64
args.bert_num_blocks = 2
args.bert_dropout = 0.1
args.bert_attn_dropout = 0.1
args.lr = 0.001
args.batch_size = 64
args.epochs = 20
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dataset_code = 'cds'

# Early stopping settings
args.patience = 1
args.primary_metric = 'NDCG@10'

ITEM_VOCAB_SIZE = 41723

# Create a dummy validation file if missing
if not os.path.exists(VALID_CSV_PATH):
    print(f"Warning: Validation file not found at '{VALID_CSV_PATH}'. Creating a dummy file.")
    dummy_df = pd.read_csv(TRAIN_CSV_PATH).sample(n=1000, random_state=42)
    dummy_df.to_csv(VALID_CSV_PATH, index=False)

# ---------------- Data loaders ----------------
train_dataset = RecommendationDataset(TRAIN_CSV_PATH, MAX_SEQUENCE_LENGTH, ITEM_VOCAB_SIZE)
valid_dataset = RecommendationDataset(VALID_CSV_PATH, MAX_SEQUENCE_LENGTH, ITEM_VOCAB_SIZE)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

# ---------------- Model & training setup ----------------
model = LRU(args).to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print(f"Using device: {args.device}")
print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

best_metric = -1.0
epochs_no_improve = 0
best_model_state = None

# ---------------- Training loop with validation & early stopping ----------------
for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

    for seqs, labels in progress_bar:
        seqs, labels = seqs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        logits, _ = model(seqs)
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss = criterion(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{args.epochs} - Train Avg Loss: {avg_epoch_loss:.4f}")

    # Validation
    print(f"Epoch {epoch+1}/{args.epochs} [Validating...]")
    metrics = evaluate(model, valid_dataloader, args.device, ks=[5, 10])
    for metric, value in metrics.items():
        print(f"  - Valid {metric}: {value:.4f}")

    # Early stopping check
    current_metric = metrics[args.primary_metric]
    if current_metric > best_metric:
        print(f"Validation metric improved from {best_metric:.4f} to {current_metric:.4f}. Saving model...")
        best_metric = current_metric
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        print(f"Validation metric did not improve for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= args.patience:
        print(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
        break

print("\nTraining finished!")

# Save best or last model
if best_model_state:
    torch.save(best_model_state, 'lru_model_best.pth')
    print("Best performing model saved to lru_model_best.pth")
else:
    torch.save(model.state_dict(), 'lru_model_last.pth')
    print("No improvement observed. Last model saved to lru_model_last.pth")

# ---------------- Load model and evaluate on test set ----------------
model = LRU(args)
model_path = 'lru_model_best.pth'
model.load_state_dict(torch.load(model_path, map_location=args.device))
model.to(args.device)
print(f"Model loaded from '{model_path}'")

test_dataset = RecommendationDataset(TEST_CSV_PATH, MAX_SEQUENCE_LENGTH, ITEM_VOCAB_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

def evaluate_test(model, dataloader, device, ks=[5, 10]):
    """
    Evaluation loop (same metrics) for test set with progress bar.
    """
    model.eval()
    hr_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for seqs, labels in progress_bar:
            seqs = seqs.to(device)
            logits, _ = model(seqs)
            scores = logits[:, -1, :]
            true_labels = labels[:, -1]
            scores.scatter_(1, seqs.to(device), -1e9)
            _, topk_indices = torch.topk(scores, k=max(ks), dim=-1)
            true_labels_expanded = true_labels.unsqueeze(-1).expand_as(topk_indices)
            hits = (topk_indices == true_labels_expanded.to(device))

            for k in ks:
                hits_k = hits[:, :k]
                hr_k = torch.any(hits_k, dim=1).float().mean().item()
                hr_scores[k].append(hr_k)
                hit_ranks = torch.where(hits_k)[1] + 1
                dcg = (1.0 / torch.log2(hit_ranks.float() + 1)).sum()
                ndcg_k = dcg / len(true_labels)
                ndcg_scores[k].append(ndcg_k.item())

    final_metrics = {}
    for k in ks:
        final_metrics[f'HR@{k}'] = np.mean(hr_scores[k])
        final_metrics[f'NDCG@{k}'] = np.mean(ndcg_scores[k])
    return final_metrics

metrics = evaluate_test(model, test_dataloader, args.device, ks=[5, 10])
print("\n---------- Evaluation Results ----------")
for metric, value in metrics.items():
    print(f"{metric:<10}: {value:.4f}")
print("----------------------------------------")
