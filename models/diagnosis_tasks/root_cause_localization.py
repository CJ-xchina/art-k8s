from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from models.unified_representation.representation import SLD, ILD, aggregate_failure_representations, aggregate_instance_representations
from evaluation import eval_RCL

# 根因定位函数（Root Cause Localization, RCL）
def RCL(model, test_samples, cases, node_dict, split_ratio=0.3, method='num', t_value=3, before=60, after=300, verbose=False):
    # 将数据集分为训练集和测试集，使用split_ratio来控制训练集和测试集的分割比例
    spilit_index = int(len(cases) * split_ratio)

    # 聚合实例级表示（ILD），用于表示每个实例的偏差
    instance_representations = aggregate_instance_representations(cases[spilit_index:], ILD(model, test_samples), before, after)

    # 聚合故障级表示（SLD），用于表示每个故障的偏差
    failure_representations, type_labels = aggregate_failure_representations(cases[spilit_index:], SLD(model, test_samples, method, t_value), None, before, after)

    # 创建一个DataFrame，保存每个故障的排名结果
    rank_df = cases[spilit_index:].reset_index(drop=True).copy(deep=True)

    # 遍历每个故障案例，为其计算根因排名
    for case_id in range(len(rank_df)):
        sld = failure_representations[case_id]  # 获取当前故障的SLD（系统级偏差）
        ilds = instance_representations[case_id]  # 获取当前故障的ILD（实例级偏差）

        # 存储每个节点与当前故障的相似度分数
        similarity_scores = []
        for (node_id, node_vector) in ilds:  # 遍历每个实例级表示中的节点
            similarity = float(cosine_similarity([node_vector], [sld]))  # 计算该节点与当前故障之间的余弦相似度
            similarity_scores.append((node_id, similarity))  # 存储节点ID及其相似度

        # 对相似度分数进行排序，按从高到低排序，得到与当前故障最相似的节点
        sorted_indices = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # 将相似度分数进行分组，按节点ID分组并计算每个节点的平均相似度
        grouped_data = defaultdict(list)
        for key, value in sorted_indices:
            grouped_data[key].append(value)

        # 计算每个节点的平均相似度
        grouped_data = {key: sum(values) / len(values) for key, values in grouped_data.items()}

        # 排序节点，得到最可能的根因节点列表
        ranks = sorted(grouped_data.items(), key=lambda x: x[1], reverse=True)

        # 格式化结果，将根因节点及其相似度分数转换为字符串格式
        ranks = [f'{node_dict[i]}:{value}' for (i, value) in ranks]

        # 将排名结果存入DataFrame的每个案例的对应列
        for i in range(len(ranks)):
            rank_df.loc[case_id, f'Top{i + 1}'] = f'{ranks[i]}'

    # 计算TopK准确率（前K个排名的准确率）和平均准确率（所有排名的平均值）
    topK, avgK = eval_RCL(rank_df, k=5, verbose=verbose)

    return rank_df, topK, avgK
