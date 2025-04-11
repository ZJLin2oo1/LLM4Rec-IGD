### Token-Level Debiasing in LLM for Recommendation: An Information Gain Approach

Our method (IGD) builds a prefix tree from the training item dataset, and applies weighted processing to SFT and Beam Search based on each token's impact on sequence entropy.

Main files:

To reproduce our method:

**For the fine-tuning stage:**

1. To implement our method and the baseline, use `cl_monitor.sh` to call `cl_monitor.py`.  
   - `beta` adjusts the weight of zero-IG tokens.  
   - `gamma` adjusts the weight of high-IG tokens (no observed effect so far, can be set to 1.0).  
   - To implement the baseline, set `beta=1.0`.  
   - For our method, `beta=0.1` works best in general. You can grid search over:  
     `[0.08, 0.1, 0.2, 0.4, 0.5, 0.6]`

2. The CFT method uses `cft_monitor.py`. According to the original paper, search over:  
   - `beta = 0.09, 0.16, 0.29, 0.38, 0.5, 0.66, 0.9, 0.96`  
   - `alpha = 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.3`

3. The Pos method is a part of the CFT method. Set `alpha=0`, and only tune `beta`.

**For the inference stage:**

4. To evaluate the performance of a single model, use `my_evaluate.sh` to call `my_evaluate.py`.  
   To run inference with multiple parameter settings at once, use `inference.py` in `inference.sh` or `inference_for_cluster`.

5. To implement inference for both the baseline and our method, adjust the `alpha` parameter in the `.sh` script:  
   - `alpha=0.0` is the baseline.  
   - In the inference script, you can set: `(0.0 0.1 0.2 0.3 0.4)`  
   - `alpha=0.2` generally yields the best results.

6. The BIGRec method uses the same model as D3. Just set the `length penalty` to `1.0` in the script.

After model training, only `model.safetensors` will be saved.  
The `tokenizer.json` will not be automatically generated in `output_dir`.  
You can find the backbone model folder under `.cache/huggingface/hub` in your home directory, then find the `snapshots` folder containing the tokenizer. Copy and paste it into `output_dir` for evaluation. You can also download it manually using `wget` from the official site.

**Details:**  
Even with a fixed `batch_size`, `minibatch_size` seems to still impact model performance. The reason is currently unclear.  
In our method, we fix `minibatch_size=16`, and it can be trained on an A100 80G GPU.  
If you lower the value, make sure to keep it consistent.

----------------------------------------------------------------------------------------


### Token-Level Debiasing in LLM for Recommendation: An Information Gain Approach

我的方法（IGD）通过train item的数据集构建一个prefix tree, 通过每一个token对序列entropy的变化来对SFT、Beam Search进行加权处理。
主要的文件：

为了复现我的方法：
对于fine-tuning阶段：
1. 要实现我的方法和baseline，请使用cl_monitor.sh调用cl_monitor.py。其中，beta是改变zero-ig token的权重。gamma改变high-ig token的权重（目前没发现效果，设置为1.0即可）。       
要实现baseline，设置beta=1.0即可。我的方法的beta一般在0.1最好，可以grid search [0.08, 0.1, 0.2, 0.4, 0.5 0.6]这些值
2. CFT方法使用cft_monitor.py。根据原文需要搜索
beta = 0.09, 0.16, 0.29, 0.38, 0.5, 0.66, 0.9, 0.96
alpha = 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.3
这些范围。
3. Pos方法是CFT方法的一部分，设置alpha=0, 只调整beta即可。

对于Inference阶段：
4. 如果只要evaluate一个model的表现，可以使用my_evaluate.sh调用my_evaluate.py。如果要一次性跑多个参数的inference结果，使用inference.py。在inference.sh或者inference_for_cluster
5. 要实现baseline的inference和我的方法，请调整.sh脚本中的alpha参数。alpha=0.0是baseline, 在inference脚本中可以设置(0.0 0.1 0.2 0.3 0.4)，一般在0.2时取得最好的效果。
6. BIGRec方法使用和D3一样的模型，只要在脚本中把length penalty调整为1.0即可。

模型训练后只会存储model.safetensor，不会在output_dir自动生成tokenizer.json可以从home下的 .cache.huggingface.hub找到你使用的backbone模型文件夹，下面的snapshots中有tokenizer, 复制粘贴到output_dir才可以进行evaluate。当然，也可以从官网下载wget下来。

一些细节：batch_size一定时，minibatch_size似乎仍会对模型性能产生影响，目前还不知道为什么。我的方法固定为16，可以使用a100 80g进行训练。若要调低，请保持一致。

----------------------------------------------------------------------------------------

下面是原本D3的Readme

### DecodingMatters

This is the raw implementation of our paper **[Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation](https://arxiv.org/abs/2406.14900)**

### Reproduce
To reproduce our results, you need to conduct the following pipeline.

```bash
# Take the book dataset as an example
# Download the dataset
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Books.json.gz
wget wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz
# Unzip
gunzip Books.json.gz
gunzip meta_Books.json.gz
# Preprocess
python ./code/preprocess.py --category "Books"
# Train
bash run.sh  # You only need to change the category parameter in script
# Inference and Evaluate
bash evaluate.sh
# Decoding Matters Inference (Our Methods) and Evaluate
bash evaluate2.sh # You need to specify your logits file in the script
```

### Results and Model
The results and the parameters of Qwen2-0.5B trained on five Amazon datasets are presented in the following table:


|Dataset|NDCG@10|HR@10|Link|
|----------------|----------------|----------------|----------------|
|CDs_and_Vinyl|0.077|0.109|[link](https://huggingface.co/USTCbaokq/BIGRec_CDs_and_Vinyl_0.5B)|
|Video_Games|0.052|0.085|[link](https://huggingface.co/USTCbaokq/BIGRec_Video_Games_0.5B)|
|Toys_and_Games|0.053|0.096|[link](https://huggingface.co/USTCbaokq/BIGRec_Toys_and_Games_0.5B)|
|Sports_and_Outdoors|0.099|0.120|[link](https://huggingface.co/USTCbaokq/BIGRec_Sports_and_Outdoors_0.5B)|
|Book|0.018|0.027|[link](https://huggingface.co/USTCbaokq/BIGRec_Books_0.5B)|


If you're using this code in your research or applications, please cite our paper using this BibTeX:
```bibtex
@article{bao2024decoding,
  title={Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation},
  author={Bao, Keqin and Zhang, Jizhi and Zhang, Yang and Huo, Xinyue and Chen, Chong and Feng, Fuli},
  journal={arXiv preprint arXiv:2406.14900},
  year={2024}
}
```
and
```bibtex
@article{bao2023bi,
  title={A bi-step grounding paradigm for large language models in recommendation systems},
  author={Bao, Keqin and Zhang, Jizhi and Wang, Wenjie and Zhang, Yang and Yang, Zhengyi and Luo, Yancheng and Chen, Chong and Feng, Fuli and Tian, Qi},
  journal={arXiv preprint arXiv:2308.08434},
  year={2023}
}
```


