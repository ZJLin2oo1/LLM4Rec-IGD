"""
Trie structure for storing tokenized item sequences with frequency and information gain (IG) scoring.

This implementation supports:
- Scoring prefixes based on frequency-weighted negative entropy
- Token-level information gain (IG) and relative IG (RIG)
- Efficient next-token score lookups
"""

import math
from typing import Dict, Tuple, List


class TrieNode:
    __slots__ = ['children', 'is_end', 'frequency', 'score', 'information_gain']

    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end: bool = False
        self.frequency: int = 0
        self.score: float = 0.0
        self.information_gain: float = 0.0


class Trie:
    def __init__(self, tokenizer, frequency_scale: float = 111806.0):
        """
        Initialize the trie.

        Args:
            tokenizer: Tokenizer with an .encode method.
            frequency_scale: A normalization constant for frequency-to-score conversion.
        """
        self.root = TrieNode()
        self.frequency_scale = frequency_scale
        self.total_frequency = 0
        self.tokenizer = tokenizer

    def insert(self, item: str, frequency: int):
        """
        Insert a tokenized item into the trie with its frequency.

        Args:
            item: The raw string of the item.
            frequency: Frequency count from training data.
        """
        tokens = self.tokenizer.encode(item, add_special_tokens=False)
        node = self.root
        norm_freq = frequency / self.frequency_scale
        item_score = -norm_freq * math.log(norm_freq)
        self.root.score += item_score

        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            node.score += item_score
            node.frequency += frequency

        node.is_end = True
        self.total_frequency += frequency

    def get_trie_score(self, sequence: str) -> float:
        """
        Get the score of a string sequence.
        """
        tokens = self.tokenizer.encode(sequence, add_special_tokens=False)
        return self.get_trie_score_by_tokens(tokens)

    def get_trie_score_by_tokens(self, tokens: List[int]) -> float:
        """
        Get the score of a tokenized sequence.
        """
        node = self.root
        for token in tokens:
            if token not in node.children:
                return float('-inf')
            node = node.children[token]
        return node.score

    def get_next_token_scores(self, tokens: List[int]) -> Tuple[float, Dict[int, float]]:
        """
        Get the score of the current sequence and all possible next-token extensions.

        Args:
            tokens: Current token sequence.

        Returns:
            current_score: Score of the current sequence.
            next_token_scores: Dict of token → score for each next possible token.
        """
        node = self.root
        for token in tokens:
            if token not in node.children:
                return float('-inf'), {}
            node = node.children[token]

        current_score = node.score
        next_token_scores = {token: child.score for token, child in node.children.items()}
        return current_score, next_token_scores

    def get_current_token_scores(self, tokens: List[int]) -> float:
        """
        Return the score of the current token sequence only.
        """
        node = self.root
        for token in tokens:
            if token not in node.children:
                return float('-inf')
            node = node.children[token]
        return node.score

    def get_last_score_difference(self, tokens: List[int]) -> float:
        """
        Get the information gain of the final token in the sequence.
        """
        if not tokens:
            return float('-inf')
        node = self.root
        for token in tokens:
            if token not in node.children:
                return float('-inf')
            node = node.children[token]
        return node.information_gain

    def compute_information_gain(self):
        """
        Compute information gain for all nodes using: IG = parent_score - child_score
        """
        stack = [(self.root, self.root.score)]
        while stack:
            node, parent_score = stack.pop()
            for child in node.children.values():
                child.information_gain = parent_score - child.score
                stack.append((child, child.score))

    def get_information_gain_statistics(self):
        """
        Return IG-based token statistics over the whole Trie.
        """
        stats = {
            'freq_zero': 0,
            'freq_small': 0,
            'freq_large': 0,
            'total_nodes': 0,
            'total_frequency': 0
        }

        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root:
                stats['total_nodes'] += 1
                stats['total_frequency'] += node.frequency
                ig = node.information_gain
                if ig == 0:
                    stats['freq_zero'] += node.frequency
                elif 0 < ig <= 3:
                    stats['freq_small'] += node.frequency
                else:
                    stats['freq_large'] += node.frequency
            stack.extend(node.children.values())
        return stats

    def get_sequence_ig(self, tokens: List[int]) -> List[float]:
        """
        Return the information gain value for each token in the sequence.

        Args:
            tokens: Tokenized input sequence.

        Returns:
            List of IG values (float) per token. Returns -inf if token path is invalid.
        """
        ig_list = []
        node = self.root
        for token in tokens:
            if token not in node.children:
                ig_list.append(float('-inf'))
                break
            node = node.children[token]
            ig_list.append(node.information_gain)
        return ig_list

    def get_sequence_rig(self, tokens: List[int]) -> List[float]:
        """
        Return the relative information gain (RIG) for each token in the sequence.

        RIG = 1 - score(child) / score(parent)

        Args:
            tokens: Tokenized input sequence.

        Returns:
            List of RIG values per token. -inf if the token path breaks.
        """
        rig_list = []
        node = self.root
        for token in tokens:
            if token not in node.children:
                rig_list.append(float('-inf'))
                break
            current_score = node.score
            node = node.children[token]
            next_score = node.score
            rig = 1 - (next_score / current_score) if current_score != 0 else 0.0
            rig_list.append(rig)
        return rig_list
