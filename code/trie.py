import math
from typing import Dict, Tuple, List


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0
        self.score = 0
        self.information_gain = 0  # 新增信息增益字段


class Trie:
    def __init__(self, tokenizer, frequency_scale=111806.0):
        self.root = TrieNode()
        self.frequency_scale = frequency_scale
        self.total_frequency = 0
        self.tokenizer = tokenizer

    def insert(self, item: str, frequency: int):
        tokens = self.tokenizer.encode(item)
        node = self.root
        item_score = (frequency / self.frequency_scale) * math.log(frequency / self.frequency_scale)

        # 直接更新 root 的 score
        self.root.score += item_score

        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            node.score += item_score
            node.frequency += frequency

        node.is_end = True
        # node.frequency = frequency
        self.total_frequency += frequency

    def get_trie_score(self, sequence: str):
        tokens = self.tokenizer.encode(sequence)
        node = self.root

        for token in tokens:
            if token not in node.children:
                return float('-inf')
            node = node.children[token]

        return node.score

    def get_trie_score_by_tokens(self, tokens):
        node = self.root
        for token in tokens:
            if token not in node.children:
                return float('-inf')
            node = node.children[token]
        return node.score

    def get_next_token_scores(self, tokens: List[int]) -> Tuple[float, Dict[int, float]]:
        """
        获取当前序列的得分以及所有可能的下一个token及其对应序列的得分

        Args:
            tokens: 当前序列的token列表

        Returns:
            tuple: (current_score, {token_id: next_sequence_score})
                - current_score: 当前序列的得分
                - Dict[token_id, score]: 所有可能的下一个token及其对应的序列得分
        """
        # 找到当前序列对应的节点
        node = self.root
        prev_node = None
        for token in tokens:
            if token not in node.children:
                return float('-inf'), {}
            prev_node = node
            node = node.children[token]

        # 获取当前序列的得分
        current_score = node.score

        # 收集所有可能的下一个token及其得分
        next_token_scores = {}
        for token, child in node.children.items():
            next_token_scores[token] = child.score

        return current_score, next_token_scores

    def get_current_token_scores(self, tokens: List[int]) -> Tuple[float, Dict[int, float]]:
        # 找到当前序列对应的节点
        node = self.root
        for token in tokens:
            if token not in node.children:
                return float('-inf'), {}
            node = node.children[token]

        # 获取当前序列的得分
        current_score = node.score
        return current_score

    def get_last_score_difference(self, tokens: List[int]) -> float:
        """
        直接查询最后一个 token 的 information_gain
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
        遍历整个 Trie，为所有节点计算 last_score - prev_score，存入 information_gain
        """

        def dfs(node, parent_score):
            for token, child in node.children.items():
                child.information_gain = child.score - parent_score
                dfs(child, child.score)

        dfs(self.root, self.root.score)

    def get_information_gain_statistics(self):
        """
        统计整个Trie中节点的信息增益分布
        Returns:
            dict: 包含不同信息增益范围的统计信息
            {
                'freq_zero': 信息增益为0的节点的频率之和,
                'freq_small': 信息增益在(0,1]范围内的节点的频率之和,
                'freq_medium': 信息增益在(1,2]范围内的节点的频率之和,
                'freq_large': 信息增益大于2的节点的频率之和,
                'total_nodes': 总节点数（不包括根节点）,
                'total_frequency': 所有节点的频率之和
            }
        """
        stats = {
            'freq_zero': 0,
            'freq_small': 0,
            'freq_large': 0,
            'total_nodes': 0,
            'total_frequency': 0
        }

        def dfs(node: TrieNode):
            if node is not self.root:  # 跳过根节点
                stats['total_nodes'] += 1
                stats['total_frequency'] += node.frequency

                ig = node.information_gain
                if ig == 0:
                    stats['freq_zero'] += node.frequency
                elif 0 < ig <= 3:
                    stats['freq_small'] += node.frequency
                else:
                    stats['freq_large'] += node.frequency

            # 递归处理所有子节点
            for child in node.children.values():
                dfs(child)

        # 从根节点开始深度优先遍历
        dfs(self.root)

        return stats

    def get_sequence_ig(self, tokens: List[int]) -> List[float]:
        """
        获取输入序列中每个 token 的 information gain 值。

        Args:
            tokens: 输入序列的 token 列表。

        Returns:
            List[float]: 每个 token 对应的 information gain 值列表。
                        如果某个 token 不在 Trie 中，则返回 float('-inf')。
        """
        ig_list = []  # 存储每个 token 的 information gain
        node = self.root  # 从根节点开始

        for token in tokens:
            if token not in node.children:
                # 如果 token 不在 Trie 中，返回 float('-inf')
                ig_list.append(float('-inf'))
                break
            node = node.children[token]
            ig_list.append(node.information_gain)  # 添加当前 token 的 information gain

        return ig_list
