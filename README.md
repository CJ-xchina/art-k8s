
# 入门指南

## 环境
建议使用 Python 3.9.13, PyTorch 1.12.1, scikit-learn 1.1.2 和 DGL 0.9.0。

## 数据集
数据集 D1 来自一个模拟的电子商务微服务系统，该系统部署在实际的云环境中，流量与真实的业务流一致。该系统由 46 个实例组成，包括 40 个微服务实例和 6 个虚拟机。通过回放故障记录收集了数据，回放时间为 2022 年 5 月的若干天。故障场景源自实际的故障类型（容器硬件、容器网络、节点 CPU、节点磁盘和节点内存相关故障）。收集到的记录被标记了各自的根本原因实例和故障类型。D1 的原始数据和标签可以在 https://anonymous.4open.science/r/Aiops-Dataset-1E0E 上公开获取。

数据集 D2 来自一家顶级商业银行的管理系统，包含 18 个实例，包括微服务、服务器、数据库和 Docker。两名经验丰富的操作员检查了 2021 年 1 月至 2021 年 6 月期间的故障记录，并标记了根本原因实例和故障类型（内存、CPU、网络、磁盘、JVM 内存和 JVM CPU 相关故障）。每位操作员分别进行了标记，并交叉检查标签以确保一致性。D2 已经在国际 AIOps 挑战赛 2021 中使用，网址：https://aiops-challenge.com。由于保密协议的原因，我们无法公开 D2 数据集。

我们已经对两个原始数据集进行了预处理，并将其放置在以下文件夹中：

D1: ART/data/D1

D2: ART/data/D2

在该文件夹中：
- `cases`：该目录下有两个文件。

  ad_cases.pkl: 故障时间戳列表。

  cases.csv: 表头中的四项分别为故障注入时间、故障级别、故障根因和故障类型。

- `hash_info`：该目录下有四个文件。它们都包含一个字典，记录了名称与索引之间的对应关系。

- `samples`：该目录下有三个文件，分别为所有样本（samples.pkl）、预训练样本（train_samples.pkl）和评估样本（test_samples.pkl）。

  每个样本是一个元组：(时间戳, 图结构, 每个节点的特征)。图结构表示微服务系统的拓扑，由调用关系和部署信息生成；每个节点的特征包括 Pod 度量特征、Pod 跟踪特征、Pod 日志特征和节点度量特征。

## 演示
我们提供了一个演示程序。
在运行以下命令之前，请先解压 D1.zip 和 D2.zip。

```
python main.py
```

## 演示中的参数说明

以数据集 D1 为例。

### 自监督学习中的预训练（SSL）

* `instance_dim`: 微服务实例的数量。（默认：46）
* `num_heads`: Transformer 编码器中的注意力头数。（默认：2）
* `tf_layers`: Transformer 编码器的层数。（默认：1）
* `channel_dim`: 数据通道数。（默认：130）
* `gnn_hidden_dim`: GraphSAGE 的隐藏层维度。（默认：64）
* `gnn_out_dim`: GraphSAGE 的输出维度。（默认：32）
* `noise_rate`: Dropout 比率。（默认：0.3）
* `gnn_layers`: GraphSAGE 的层数。（默认：2）
* `gru_hidden_dim`: GRU 的隐藏层维度。（默认：32）
* `gru_layers`: GRU 的层数。（默认：1）
* `epochs`: 训练的轮数。（默认：1000）
* `batch_size`: 批处理大小。（默认：4）
* `learning_rate`: 学习率。（默认：0.001）

### 下游任务

* `split_ratio`: 初始化集和测试集的划分比例。（默认：0.6）
* `method & t_value`: 聚合 ILD 到 SLD 的方法和参数。方法可以是 'num' 或 'prob'，分别表示指定数量和概率方法。'num' 可以取值为 [1, instance_dim] 范围内的整数，'prob' 可以取值为 [0,1) 范围内的浮动值。（默认：'num', 3）
* `q & level`: SPOT 初始化参数。（默认：0.1, 0.95）
* `delay`: 将小于延迟时间间隔的异常视为同一故障。（默认：600）
* `impact_window`: 单次故障的影响范围。与 'before' 和 'after' 类似。（默认：300）
* `before & after`: 故障注入时间为 inject_ts。假设故障影响范围为 [inject_ts-before, inject_ts+after]，在此时间窗口内的数据将用于分析。（默认：59, 300）
* `max_clusters`: 从割树中获得的最大聚类数。（默认：25）
* `verbose`: 控制输出的详细程度。（默认：False）

更多细节可以在配置文件中找到：ART/config/D1.yaml。
```
