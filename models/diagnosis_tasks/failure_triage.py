import copy
from collections import Counter

import numpy as np
from models.unified_representation.representation import SLD, aggregate_failure_representations
from sklearn.metrics.pairwise import cosine_similarity

from evaluation import eval_FT


# 定义故障信息类，用于存储每个故障实例的向量表示和标签
class FailureInfo:
    def __init__(self, vector, label):
        self.vector = vector  # 向量表示（系统的特征）
        self.label = label  # 故障的标签（标记故障类型）


# 定义节点类，用于构建故障分类树
class Node:
    cluster_id = -1  # 用于标识当前节点的聚类ID
    common_split_dims = set()  # 存储用于分割的维度，避免重复

    def __init__(self, failure_infos, depth):
        self.failure_infos = failure_infos  # 当前节点的故障信息集合
        self.depth = depth  # 当前节点的深度
        self.left = None  # 左子树
        self.right = None  # 右子树
        self.flag = 1  # 用于控制是否继续分割
        self.split_value = None  # 分割值
        self.cluster_id = -1  # 聚类ID
        self.update_criteria()  # 更新分割准则
        self.update_label_id()  # 更新标签ID

    # 更新分割准则，选择方差最大的维度进行分割
    def update_criteria(self):
        if self.flag:  # 如果当前节点可以分割
            vectors = np.array([failure_info.vector for failure_info in self.failure_infos])
            variances = np.var(vectors, axis=0)  # 计算各维度的方差
            for dim in Node.common_split_dims:  # 如果该维度已经被使用过，则将其方差设为0
                variances[dim] = 0
            split_dim = np.argmax(variances)  # 选择方差最大的维度进行分割
            criteria = variances[split_dim]  # 分割的标准（方差）
            self.split_dim, self.criteria = split_dim, criteria  # 设置分割维度和标准

    # 更新标签ID，选择出现频率最高的标签作为当前节点的标签
    def update_label_id(self):
        label_counts = Counter([failure_info.label for failure_info in self.failure_infos])  # 统计每个标签的出现次数
        most_common_label = max(label_counts, key=label_counts.get)  # 选择频率最高的标签
        self.label_id = most_common_label  # 设置当前节点的标签ID

    # 更新节点的内部距离，基于余弦相似度来计算
    def update_in_distance(self):
        vectors = [info.vector for info in self.failure_infos]
        self.in_distance = np.mean(1 - cosine_similarity(vectors))  # 计算所有向量的平均余弦相似度，作为内部距离


def split_cluster(root, max_depth=50, min_cluster_size=1):
    # 初始化叶节点列表，从根节点开始
    leaf_nodes = [root]

    while leaf_nodes:  # 当还有叶节点可以分割时
        # 从当前叶节点中选择“方差”最大的节点进行分割，方差越大，数据的变化越明显
        max_criteria_node = max(leaf_nodes, key=lambda x: x.criteria)  # 选择方差最大节点，x.criteria 是 x 节点的最大方差

        split_dim = max_criteria_node.split_dim  # 获取当前节点分割的维度
        # 获取当前节点的所有故障信息的向量表示
        vectors = np.array([failure_info.vector for failure_info in max_criteria_node.failure_infos])

        max_cosine_distance = -1  # 初始化最大余弦距离
        best_percentile = None  # 初始化最佳分割点

        # 遍历当前节点在分割维度上的所有值，寻找最佳的分割点
        for percentile in vectors[:, split_dim]:  # 对每个维度值进行遍历
            # 根据当前分割值，将数据分为左右两部分
            left_failure_infos = [failure_info for failure_info in max_criteria_node.failure_infos if
                                  failure_info.vector[split_dim] <= percentile]  # 左侧子集
            right_failure_infos = [failure_info for failure_info in max_criteria_node.failure_infos if
                                   failure_info.vector[split_dim] > percentile]  # 右侧子集

            # 判断分割后的左右子集是否满足最小聚类大小要求
            if len(left_failure_infos) >= min_cluster_size and len(right_failure_infos) >= min_cluster_size:
                # 如果满足条件，计算左右子集之间的余弦相似度
                left_vectors = np.array([failure_info.vector for failure_info in left_failure_infos])
                right_vectors = np.array([failure_info.vector for failure_info in right_failure_infos])
                cosine_distance = np.mean(1 - cosine_similarity(left_vectors, right_vectors))  # 计算余弦距离

                # 如果当前分割点使得余弦距离最大，则更新最佳分割点
                if cosine_distance > max_cosine_distance:
                    max_cosine_distance = cosine_distance
                    best_percentile = percentile  # 更新最佳分割点

        # 如果找到有效的最佳分割点
        if best_percentile is not None:
            max_criteria_node.split_value = best_percentile  # 设置当前节点的分割点

            # 根据最佳分割点，重新划分当前节点的故障信息为左子集和右子集
            left_failure_infos = [failure_info for failure_info in max_criteria_node.failure_infos if
                                  failure_info.vector[split_dim] <= best_percentile]
            right_failure_infos = [failure_info for failure_info in max_criteria_node.failure_infos if
                                   failure_info.vector[split_dim] > best_percentile]

            # 创建左子树和右子树，递归分割
            max_criteria_node.left = Node(left_failure_infos, max_criteria_node.depth + 1)
            max_criteria_node.right = Node(right_failure_infos, max_criteria_node.depth + 1)

        # 当前节点分割完毕，标记为不可再分割
        max_criteria_node.flag = 0

        # 更新当前节点的“内部距离”属性（子集之间的平均余弦距离）
        max_criteria_node.update_in_distance()

        # 将当前维度加入已经使用的维度集合，避免重复使用相同的维度进行分割
        Node.common_split_dims.add(split_dim)

        # 获取仍然可以继续分割的叶节点，限制最大深度，避免树的深度过大
        leaf_nodes = [node for node in get_leaf_nodes(root) if ((node.depth < max_depth) & node.flag)]

    # 对所有叶节点进行聚类ID的分配
    for leaf_node in get_leaf_nodes(root):
        Node.cluster_id += 1  # 聚类ID递增
        leaf_node.cluster_id = Node.cluster_id  # 分配当前聚类ID


# 合并节点，直到达到最大聚类数
def merge_nodes(root, max_clusters):
    while len(get_leaf_nodes(root)) >= (max_clusters + 1):
        min_avg_cosine_distance = float('inf')
        node_to_merge = None
        for node in get_parent_nodes_of_leaves(root):
            if node.in_distance < min_avg_cosine_distance:
                min_avg_cosine_distance = node.in_distance
                node_to_merge = node
        if node_to_merge is not None:
            node_to_merge.left = None
            node_to_merge.right = None
        else:
            break


# 获取所有叶节点
def get_leaf_nodes(node):
    if node.left is None and node.right is None:
        node.update_criteria()
        return [node]
    leaf_nodes = []
    if node.left is not None:
        leaf_nodes.extend(get_leaf_nodes(node.left))
    if node.right is not None:
        leaf_nodes.extend(get_leaf_nodes(node.right))
    return leaf_nodes


# 获取所有叶节点的父节点
def get_parent_nodes_of_leaves(root):
    parent_nodes = set()
    leaf_nodes = get_leaf_nodes(root)
    for node in leaf_nodes:
        parent_node = find_parent(root, node)
        if parent_node is not None:
            parent_nodes.add(parent_node)
    return list(parent_nodes)


# 查找父节点
def find_parent(root, node):
    if root is None:
        return None
    if root.left == node or root.right == node:
        return root
    left_parent = find_parent(root.left, node)
    right_parent = find_parent(root.right, node)
    return left_parent if left_parent is not None else right_parent


# 初始化预测标签和聚类
def init_prediction(root, type_hash):
    init_labels, pre, clusters = [], [], []
    leaf_nodes = get_leaf_nodes(root)
    for leaf_node in leaf_nodes:
        for info in leaf_node.failure_infos:
            init_labels.append(type_hash[info.label])  # 将标签映射为类型
            pre.append(type_hash[leaf_node.label_id])  # 使用节点的标签ID进行预测
            clusters.append(leaf_node.cluster_id)  # 保存聚类ID
    return init_labels, pre, clusters


# 测试预测结果
def test_prediction(root, test_failure_infos, type_dict):
    pre, clusters = [], []
    for test_failure_info in test_failure_infos:
        current_node = root
        # 遍历树结构，找到最终的叶节点进行预测
        while current_node.left is not None or current_node.right is not None:
            if test_failure_info.vector[current_node.split_dim] <= current_node.split_value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        pre_id, cluster_id = current_node.label_id, current_node.cluster_id
        pre.append({v: k for k, v in type_dict.items()}[pre_id])  # 将预测ID转换为标签
        clusters.append(cluster_id)  # 获取聚类ID
    return pre, clusters


# 故障分类函数
def FT(model, test_samples, cases, type_hash, type_dict, split_ratio=0.7, method='num', t_value=3, before=60, after=300,
       max_clusters=15, channel_dict=None, verbose=False):
    # 获取故障表示（失败表示）及其标签
    failure_representations, type_labels = aggregate_failure_representations(cases,
                                                                             SLD(model, test_samples, method, t_value),
                                                                             type_hash, before, after)

    # 根据分割比例，将数据集分为训练集和测试集
    spilit_index = int(len(cases) * split_ratio)
    init_failure_infos = [FailureInfo(failure_representations[_], type_dict[type_labels[_]]) for _ in
                          range(spilit_index)]
    test_failure_infos = [FailureInfo(failure_representations[_], type_dict[type_labels[_]]) for _ in
                          range(spilit_index, len(failure_representations))]
    test_labels = [type_labels[_] for _ in range(spilit_index, len(failure_representations))]

    # 创建树并进行分割
    Node.cluster_id = -1
    Node.common_split_dims = set()
    splitting_root = Node(init_failure_infos, depth=0)
    split_cluster(splitting_root, max_depth=50, min_cluster_size=1)

    # 进行节点合并，直到达到最大聚类数
    merged_root = copy.deepcopy(splitting_root)
    merge_nodes(merged_root, max_clusters)

    # 评估初始化结果
    num_leaf_nodes = len(get_leaf_nodes(merged_root))
    if verbose:
        init_labels, init_pre, init_clusters = init_prediction(merged_root, {v: k for k, v in type_dict.items()})
        print('init_prediction: ', end='')
        eval_FT(merged_root, init_labels, init_pre, num_leaf_nodes, verbose=True)

    # 评估测试集结果
    test_pre, test_clusters = test_prediction(merged_root, test_failure_infos, type_dict)
    if verbose:
        print('test_prediction: ', end='')
    precision, recall, f1 = eval_FT(merged_root, test_labels, test_pre, num_leaf_nodes, channel_dict, verbose)
    pre_types = [type_dict[item] for item in test_pre]
    return pre_types, precision, recall, f1
