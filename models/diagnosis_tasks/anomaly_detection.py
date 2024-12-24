import numpy as np
from models.unified_representation.representation import SLD  # 导入SLD（统一表示）模型，用于计算异常分数
from utils.spot import SPOT  # 导入SPOT（自适应分位点检验）模型，用于异常阈值计算

from evaluation import eval_AD  # 导入评估函数，用于评估异常检测的效果


# 定义函数run_spot，用于通过SPOT模型计算异常的极值分位数，论文中的EVT阈值
def run_spot(his_val, q=1e-2, level=0.98, verbose=False):
    model = SPOT(q)  # 创建SPOT模型实例，q为模型参数，表示异常检测的灵敏度
    model.fit(his_val, [])  # 使用历史数据（his_val）拟合SPOT模型
    model.initialize(level=level, verbose=verbose)  # 初始化模型，设置分位点和是否打印详细过程
    return model.extreme_quantile  # 返回极值分位点（用于判断异常的阈值）


# 定义函数get_threshold，用于获取用于异常检测的阈值
def get_threshold(his_val, q=1e-2, level=0.98, verbose=False):
    try:
        if len(set(his_val)) == 1:  # 如果所有历史数据值相同，则直接将阈值设为该值
            threshold = his_val[0]
        else:
            threshold = run_spot(his_val, q, level, verbose)  # 使用SPOT模型计算阈值
    except:  # 如果计算SPOT时发生错误，使用简单的排序方法获取阈值
        threshold = sorted(his_val)[int(len(his_val) * level)]  # 使用分位点来获取阈值
    return threshold  # 返回计算得到的阈值


# 定义函数get_pre_interval，用于计算并获取异常时间区间
def get_pre_interval(eval_slds_sum, delay=600):
    pre_ts_df = eval_slds_sum[eval_slds_sum['outlier'] == 1]  # 筛选出标记为异常的时间点
    pre_ts_df['diff'] = [0] + np.diff(pre_ts_df['timestamp']).tolist()  # 计算相邻时间点的差异，表示时间间隔
    pre_interval = []  # 用于存储异常区间
    start_ts, end_ts = None, None  # 初始化起始时间和结束时间
    for _, span in pre_ts_df.iterrows():  # 遍历异常时间点
        if start_ts is None:  # 如果起始时间为空，则设置为当前时间点
            start_ts = int(span['timestamp'])
        if span['diff'] >= delay:  # 如果两个异常点之间的间隔超过了延迟阈值，则视为一个新的异常区间
            pre_interval.append((start_ts, end_ts))  # 添加当前异常区间
            start_ts = int(span['timestamp'])  # 更新起始时间
        end_ts = int(span['timestamp'])  # 更新结束时间
    pre_interval.append((start_ts, end_ts))  # 最后一段异常区间
    # 过滤掉起始时间和结束时间相同的异常区间
    pre_interval = [(item[0], item[1]) for item in pre_interval if item[0] != item[1]]
    return pre_interval  # 返回异常区间列表


# 定义异常检测主函数AD
def AD(model, train_samples, test_samples, ad_cases_label, split_ratio=0.6, method='num', t_value=3, q=0.1, level=0.95,
       delay=600, impact_window=300, verbose=False):
    # 将训练和测试样本合并，并按照时间戳排序
    total_samples = train_samples + test_samples
    total_samples.sort(key=lambda x: x[0])  # 按时间戳升序排列
    ts_list = sorted([item[0] for item in total_samples])  # 提取所有时间戳并排序
    split_ts = ts_list[int(len(ts_list) * split_ratio)]  # 根据split_ratio确定训练和测试集的分割点

    # 筛选测试集中的异常标签（ad_cases_label）在测试集时间戳之后的数据
    ad_cases_label = [item for item in ad_cases_label if item > split_ts]

    # 初始化训练和测试数据集
    init_samples = [item for item in train_samples if item[0] <= split_ts]  # 训练集样本
    eval_samples = [item for item in total_samples if item[0] > split_ts]  # 测试集样本

    # 计算训练集和测试集的SLD（统一表示）
    init_slds = SLD(model, init_samples, method, t_value)  # 训练集的统一表示
    eval_slds = SLD(model, eval_samples, method, t_value)  # 测试集的统一表示

    # 对训练集和测试集的SLD结果进行汇总
    init_slds_sum = init_slds.set_index('timestamp').sum(axis=1).reset_index()  # 汇总训练集SLD L1结果
    eval_slds_sum = eval_slds.set_index('timestamp').sum(axis=1).reset_index()  # 汇总测试集SLD L1结果

    # 计算异常检测的阈值
    threshold = get_threshold(init_slds_sum[0].values, q, level)  # 获取阈值

    if threshold > 0:  # 如果阈值有效
        # 标记测试集中的异常点（outlier标记为1）
        eval_slds_sum['outlier'] = eval_slds_sum[0].apply(lambda x: 1 if x > threshold else 0)

        if verbose:  # 如果需要详细信息，打印阈值和异常比例
            print(f'threshold is {threshold}.')
            print(f"outlier ratio is {np.round(sum(eval_slds_sum['outlier'] == 1) / len(eval_slds_sum), 4)}.")

        # 获取异常时间区间
        pre_interval = get_pre_interval(eval_slds_sum, delay)

        # 使用评估函数计算并返回精确度、召回率和F1分数
        precision, recall, f1 = eval_AD(pre_interval, ad_cases_label, impact_window, verbose)
        return pre_interval, precision, recall, f1  # 返回异常时间区间和评估指标
    else:
        print('threshold is invalid.')  # 如果阈值无效，则输出错误信息
