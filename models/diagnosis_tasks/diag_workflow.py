'''
a workflow for multiple diagnosis tasks
'''

import sys
import os

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


from diagnosis_tasks.failure_triage import FT
from diagnosis_tasks.root_cause_localization import RCL

from anomaly_detection import AD


def diag_workflow(config, model, train_samples, test_samples,
                  cases, ad_cases_label,
                  node_dict, type_hash, type_dict, channel_dict=None,
                  workflow=['AD', 'FT', 'RCL']):
    tmp_res, eval_res = {}, {}

    # 异常检测任务 (Anomaly Detection - AD)
    # 如果工作流中包含'AD'任务，则执行以下异常检测步骤。
    if 'AD' in workflow:
        print('AD start. |', '*' * 100)

        # 从配置文件中提取与异常检测相关的参数。
        tmp_param = config['AD']
        split_ratio = tmp_param['split_ratio']  # 数据集的训练与测试分割比例。
        method = tmp_param['method']  # 异常检测使用的方法。
        t_value = tmp_param['t_value']  # 时间窗口参数。
        q = tmp_param['q']  # 用于选择异常实例的阈值。
        level = tmp_param['level']  # 异常检测的粒度（如实例级或系统级）。
        delay = tmp_param['delay']  # 设置延迟窗口以过滤偶然异常。
        impact_window = tmp_param['impact_window']  # 分析异常影响的时间窗口。
        verbose = tmp_param['verbose']  # 是否显示详细过程信息。

        # 调用 AD 函数执行异常检测，并返回以下结果：
        # - `pre_interval`: 异常时间区间。
        # - `precision`: 精确率，评估检测结果的准确性。
        # - `recall`: 召回率，评估异常检测的覆盖程度。
        # - `f1`: 综合了精确率与召回率的 F1 分数。
        pre_interval, precision, recall, f1 = AD(model, train_samples, test_samples, ad_cases_label,
                                                 split_ratio, method, t_value, q, level, delay, impact_window, verbose)

        # 保存检测结果和评估指标。
        tmp_res['AD'] = {'pre_interval': pre_interval}  # 异常时间区间。
        eval_res['AD'] = {'precision': precision, 'recall': recall, 'f1': f1}  # 评估指标。

        # 输出异常检测完成的信息。
        print('AD done. | ', eval_res['AD'])

    # failure triage & interpretable channel details
    # 故障分类与可解释的通道细节（Failure Triage & Interpretable Channel Details）
    if 'FT' in workflow:  # 如果工作流中包括'FT'（故障分类）任务，则执行以下代码
        print('FT start. |', '*' * 100)  # 输出开始故障分类任务的标识

        # 从配置文件中读取与故障分类（FT）相关的参数
        tmp_param = config['FT']
        split_ratio = tmp_param['split_ratio']  # 数据集的训练与测试分割比例
        method = tmp_param['method']  # 使用的故障分类方法
        t_value = tmp_param['t_value']  # 用于计算异常度量的阈值
        before = tmp_param['before']  # 分析异常前的时间窗口
        after = tmp_param['after']  # 分析异常后的时间窗口
        max_clusters = tmp_param['max_clusters']  # 故障类型的最大聚类数
        verbose = tmp_param['verbose']  # 是否打印详细的过程信息

        # 调用FT函数执行故障分类任务，传入模型、测试样本、案例等信息
        # FT函数返回三个结果：预测的故障类型、精确度（precision）、召回率（recall）、F1分数
        pre_types, precision, recall, f1 = FT(
            model, test_samples, cases, type_hash, type_dict, split_ratio, method, t_value,
            before, after, max_clusters, channel_dict=channel_dict, verbose=verbose
        )

        # 将故障分类的结果（预测的类型）保存到tmp_res字典中
        tmp_res['FT'] = {'pre_types': pre_types}
        # 将评估结果（精确度、召回率、F1分数）保存到eval_res字典中
        eval_res['FT'] = {'precision': precision, 'recall': recall, 'f1': f1}

        # 输出故障分类任务的完成信息及其评估结果
        print('FT done. |', eval_res['FT'])

    # 根因定位（RCL）部分：用于定位故障的根本原因
    if 'RCL' in workflow:  # 如果工作流中包括根因定位（RCL）任务，则执行以下代码
        print('RCL start. |', '*' * 100)  # 打印根因定位开始的标识，方便跟踪任务进度

        # 从配置文件中获取与根因定位（RCL）相关的参数
        tmp_param = config['RCL']
        split_ratio = tmp_param['split_ratio']  # 控制训练集与测试集的分割比例（通常是0.7或0.8）
        method = tmp_param['method']  # 使用的计算方法，可能是“num”或其他方式
        t_value = tmp_param['t_value']  # 用于计算阈值的T值，通常用于确定异常的度量标准
        before = tmp_param['before']  # 用于计算根因定位时故障前的时间窗口
        after = tmp_param['after']  # 用于计算根因定位时故障后的时间窗口
        verbose = tmp_param['verbose']  # 是否打印详细的调试信息

        # 调用RCL函数执行根因定位任务
        # RCL函数的作用是计算每个测试样本的根因排名，并返回根因定位的排名结果
        # 返回的结果包括：
        # - rank_df: 包含每个故障的根因排名结果的DataFrame
        # - topK: 前K个排名的准确率
        # - avgK: 平均准确率
        rank_df, topK, avgK = RCL(
            model, test_samples, cases, node_dict,
            split_ratio, method, t_value, before, after, verbose
        )

        # 将根因定位的排名结果存储到tmp_res字典中，方便后续使用
        tmp_res['RCL'] = {'rank_df': rank_df}

        # 将评估结果（TopK准确率和平均准确率）存储到eval_res字典中
        eval_res['RCL'] = {'topK': topK, 'avgK': avgK}

        # 打印根因定位任务完成的标识，并显示评估结果（TopK和avgK）
        print('RCL done. |', eval_res['RCL'])

    return tmp_res, eval_res
