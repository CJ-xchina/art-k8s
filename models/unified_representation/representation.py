import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from layers import create_dataloader_AR  # 用于创建数据加载器


# method: 'num', 'prob'
# SLD (System-Level Deviation): 计算系统级偏差
def SLD(model, test_samples, method='num', t_value=3):
    mse = nn.MSELoss(reduction='none')  # 使用均方误差（MSE）作为损失函数
    system_level_deviation_df = pd.DataFrame()  # 用于存储计算的系统级偏差结果
    dataloader = create_dataloader_AR(test_samples, batch_size=128, shuffle=False)  # 创建数据加载器
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不需要梯度计算
        # 遍历数据加载器中的每个批次
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
            z, h = model(batched_graphs, batched_feats)  # 前向传播，得到模型的输出
            loss = mse(h, batched_targets)  # 计算损失（每个样本的误差）

            # 根据方法选择不同的计算方式
            if method == 'prob':
                max = torch.max(torch.sum(loss, dim=-1), dim=-1).values.unsqueeze(dim=-1)
                min = torch.min(torch.sum(loss, dim=-1), dim=-1).values.unsqueeze(dim=-1)
                root_prob = torch.softmax((torch.sum(loss, dim=-1) - min) / (max - min), dim=-1)  # 计算根因概率

                sorted_indices = torch.argsort(root_prob, dim=1, descending=True)  # 按概率降序排列
                root_prob = torch.gather(root_prob, 1, sorted_indices)  # 获取排序后的根因概率
                loss = torch.gather(loss, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, loss.size(-1)))  # 获取排序后的损失
                cumulative_sum = torch.cumsum(root_prob, dim=1)  # 计算累积概率

                # 选择达到阈值的根因
                t_value_indices = torch.argmax((cumulative_sum > t_value).to(torch.int), dim=1)
                selected_indices = torch.zeros_like(loss)  # 初始化选择的索引
                for i in range(root_prob.shape[0]):
                    selected_indices[i, :t_value_indices[i] + 1, ] = 1  # 选择前t_value个根因
                system_level_deviation = torch.sum(selected_indices * loss, dim=1)  # 计算最终的系统级偏差
            elif method == 'num':
                instance_deviation = torch.sum(loss, dim=-1)  # 计算实例级偏差
                topk_values, topk_indices = torch.topk(instance_deviation, k=t_value, dim=-1)  # 选择前t_value个偏差最大的实例
                mask = torch.zeros_like(instance_deviation)
                mask = mask.scatter_(1, topk_indices, 1).unsqueeze(-1)  # 根据索引将mask中前t_value个位置置为1
                system_level_deviation = torch.sum(loss * mask, dim=1)  # 计算选中的根因的偏差

            tmp_df = pd.DataFrame(system_level_deviation.detach().numpy())  # 将计算结果转为DataFrame
            tmp_df['timestamp'] = batch_ts  # 添加时间戳列
            system_level_deviation_df = pd.concat([system_level_deviation_df, tmp_df])  # 将结果合并

    # 返回系统级偏差的DataFrame
    return system_level_deviation_df.reset_index(drop=True)

# ILD (Instance-Level Deviation): 计算实例级偏差
def ILD(model, test_samples):
    mse = nn.MSELoss(reduction='none')  # 使用均方误差作为损失函数
    instance_level_deviation_df = pd.DataFrame()  # 用于存储实例级偏差结果
    dataloader = create_dataloader_AR(test_samples, batch_size=128, shuffle=False)  # 创建数据加载器
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不需要梯度计算
        # 遍历数据加载器中的每个批次
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
            z, h = model(batched_graphs, batched_feats)  # 前向传播，得到模型的输出
            loss = mse(h, batched_targets)  # 计算损失（每个样本的误差）
            batch_size, instance_size, channel_size = loss.shape
            # 将损失值转换为字符串形式，便于存储
            string_tensor = np.array([str(row.tolist()) for row in loss.reshape(-1, channel_size)])
            tmp_df = pd.DataFrame(string_tensor.reshape(batch_size, instance_size))  # 将损失转换为DataFrame
            tmp_df['timestamp'] = batch_ts  # 添加时间戳列
            instance_level_deviation_df = pd.concat([instance_level_deviation_df, tmp_df])  # 将结果合并

    # 返回实例级偏差的DataFrame
    return instance_level_deviation_df.reset_index(drop=True)

# 聚合实例级偏差表示（实例级的度量）
def aggregate_instance_representations(cases, instance_level_deviation_df, before=60, after=300):
    instance_representations = []
    for _, case in cases.iterrows():
        instance_representation = []
        # 获取指定时间窗口内的实例级偏差数据
        agg_df = instance_level_deviation_df[
            (instance_level_deviation_df['timestamp'] >= (case['timestamp'] - before)) &
            (instance_level_deviation_df['timestamp'] < (case['timestamp'] + after))]

        # 聚合每一列数据，生成实例级表示
        for col_name, col_data in agg_df.items():
            if col_name == 'timestamp':
                continue
            instance_representation.extend([(col_name, eval(item)) for item in col_data])  # 不进行聚合，直接扩展数据

        instance_representations.append(instance_representation)  # 添加该实例的表示

    return instance_representations

# 聚合故障级偏差表示（系统级的度量）
def aggregate_failure_representations(cases, system_level_deviation_df, type_hash=None, before=60, after=300):
    failure_representations, type_labels = [], []
    for _, case in cases.iterrows():
        # 获取指定时间窗口内的系统级偏差数据
        agg_df = system_level_deviation_df[(system_level_deviation_df['timestamp'] >= (case['timestamp'] - before)) &
                                           (system_level_deviation_df['timestamp'] < (case['timestamp'] + after))]
        failure_representations.append(list(agg_df.mean()[:-1]))  # 计算该窗口内的平均值
        if type_hash:
            type_labels.append(type_hash[case['failure_type']])  # 获取故障类型标签
        else:
            type_labels.append(case['failure_type'])  # 如果没有提供type_hash，则直接使用故障类型列

    return failure_representations, type_labels
