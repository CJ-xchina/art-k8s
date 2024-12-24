from collections import Counter

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


# 异常检测（Anomaly Detection, AD）的评估函数
def eval_AD(pre_interval, ad_cases_label, impact_window=5 * 60, verbose=False):
    # pre_interval: 预测的异常时间区间列表
    # ad_cases_label: 真实的异常标签时间戳列表
    # impact_window: 定义异常影响范围的窗口（默认为5分钟）

    # 初始化字典：pre_dict 存储预测的异常区间，ad_dict 存储实际的异常标签
    pre_dict = {key: set() for key in pre_interval}
    ad_dict = {key: set() for key in ad_cases_label}

    # 逐个计算预测的异常区间和实际的异常标签时间戳是否重叠
    for s, e in pre_interval:  # s, e 为预测的异常区间（开始时间，结束时间）
        for case_ts in ad_cases_label:  # 遍历每个实际的异常标签
            case_s, case_e = case_ts - impact_window, case_ts + impact_window  # 将每个异常标签的影响范围扩展
            # 如果实际异常标签与预测异常区间重叠，进行匹配
            if case_s <= e and case_e >= s:
                pre_dict[(s, e)].add(case_ts)  # 将实际异常标签加入到预测区间中
                ad_dict[case_ts].add((s, e))  # 将预测区间加入到实际异常标签中

    # 计算评价指标：TP（真正例）、FP（假正例）、FN（假负例）
    TP = len([key for key, value in ad_dict.items() if len(value) > 0])  # 计算真正例（实际异常标签与预测区间匹配）
    FP = len([key for key, value in pre_dict.items() if len(value) == 0])  # 计算假正例（预测的异常区间没有匹配的实际标签）
    FN = len([key for key, value in ad_dict.items() if len(value) == 0])  # 计算假负例（实际异常标签没有匹配的预测区间）

    # 计算精确度（precision）、召回率（recall）、F1值
    precision = np.round(TP / (TP + FP), 4)  # 精确度
    recall = np.round(TP / (TP + FN), 4)  # 召回率
    f1 = np.round(2 * precision * recall / (precision + recall), 4)  # F1值

    # 计算密度：预测区间中包含异常标签的比例
    density = np.round(np.mean([len(value) for key, value in pre_dict.items() if len(value) > 0]), 2)

    # 如果 verbose 为 True，输出详细的评估信息
    if verbose:
        print(f'precision: {precision}, recall: {recall}, f1: {f1}, density: {density}')

    return precision, recall, f1  # 返回精确度、召回率和F1值


# 故障分类（Failure Triage, FT）的评估函数
def eval_FT(root, labels, pre, num_leaf_nodes, channel_dict=None, verbose=False):
    avg_type = 'weighted'  # 使用加权平均计算F1值

    # 计算加权精确度、召回率和F1值
    precision = np.round(precision_score(labels, pre, average=avg_type), 4)
    recall = np.round(recall_score(labels, pre, average=avg_type), 4)
    f1 = np.round(f1_score(labels, pre, average=avg_type), 4)

    # 计算标签率：叶节点数与总故障信息数的比例
    label_rate = np.round(num_leaf_nodes / len(root.failure_infos), 4)

    # 如果 verbose 为 True，输出详细的评估信息
    if verbose:
        print(f'precision, recall, {avg_type}-f1, label_rate')
        print(precision, recall, f1, label_rate, f'({num_leaf_nodes} / {len(root.failure_infos)})')

    # 如果提供了 channel_dict，则打印每个通道的详细信息
    if channel_dict:
        print_channel_detials(root, channel_dict)

    return precision, recall, f1  # 返回精确度、召回率和F1值


# 打印每个通道的详细信息
def print_channel_detials(node, channel_dict):
    if node is None:
        return
    indent = '     ' * node.depth  # 根据节点的深度增加缩进
    # 如果是叶节点（没有左右子节点），则打印该节点的详细信息
    if node.left is None and node.right is None:
        print(
            indent + f' | Split Dimension: {node.split_dim}, Split Criteria: {node.criteria}, Split Value: {node.split_value}, Num Vectors: {len(node.failure_infos)}, In distance: {node.in_distance}')
        print(indent + ' * ' + f'[{node.label_id}] ' + str(
            Counter([failure_info.label for failure_info in node.failure_infos])))  # 输出每个标签的频数
    else:
        # 如果是非叶节点，则打印分割维度和标准，并附带通道名称
        print(
            indent + f'Split Dimension: {node.split_dim}, Split Criteria: {node.criteria}, Split Value: {node.split_value}, Num Vectors: {len(node.failure_infos)}, In distance: {node.in_distance}, {channel_dict[node.split_dim]}')

    # 递归打印左子树和右子树的详细信息
    print_channel_detials(node.left, channel_dict)
    print_channel_detials(node.right, channel_dict)


# 根因定位（Root Cause Localization, RCL）的评估函数
def eval_RCL(rank_df, k=5, verbose=False):
    topks = np.zeros(k)  # 初始化一个长度为k的数组，用于记录前K个位置的匹配情况

    # 遍历每个案例，检查是否在前K个预测位置中出现了正确的根因实例
    for _, case in rank_df.iterrows():
        for i in range(k):
            if case['cmdb_id'] in case[f'Top{i + 1}']:  # 如果根因实例在TopK列表中，则从当前位置开始累加
                topks[i:] += 1  # 将该位置及其之后的位置累加
                break

    # 计算TopK的平均值和各个位置的比例
    topK = list(np.round(topks / len(rank_df), 4))  # 计算每个TopK位置的匹配比例
    avgK = np.round(np.mean(topks / len(rank_df)), 4)  # 计算平均匹配比例

    # 如果 verbose 为 True，输出详细的TopK信息
    if verbose:
        print(f'topK: {topK}, avgK: {avgK}')

    return topK, avgK  # 返回TopK和平均TopK的匹配比例
