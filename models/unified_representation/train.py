import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from layers import AutoRegressor, create_dataloader_AR


def train(samples, config):
    # hyperparameters
    batch_size = config['batch_size']
    epochs = config['epochs']

    tf_in_dim = config['instance_dim']
    num_heads = config['num_heads']
    tf_layers = config['tf_layers']

    gnn_in_dim = config['channel_dim']
    gnn_hidden_dim = config['gnn_hidden_dim']
    gnn_out_dim = config['gnn_out_dim']
    dropout = config['noise_rate']
    gnn_layers = config['gnn_layers']

    gru_hidden_dim = config['gru_hidden_dim']
    gru_layers = config['gru_layers']

    learning_win_size = config['learning_win_size']
    learning_max_gap = config['learning_max_gap']


    PATIENCE = 5
    early_stop_threshold = 1e-3
    prev_loss = np.inf
    stop_count = 0

    # model init
    model = AutoRegressor(tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout,
                          tf_layers, gnn_layers, gru_layers)
    best_state_dict = model.state_dict()
    Loss = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)

    # 搞糟用于自动回归模型（AutoRegressor）的数据加载器
    dataloader = create_dataloader_AR(samples, window_size=learning_win_size, max_gap=learning_max_gap,
                                      batch_size=batch_size,
                                      shuffle=True)
    for epoch in tqdm(range(epochs)):
        running_loss = []
        # batch_ts (64,) batch_size 个时间戳
        # batched_graph 64 个图
        # batched_targets (64, 46, 130) 也就是 (batch_size, node_num, features)
        # batched_feats (64, 5, 46, 130) 也就是 (batch_size, step_num, node_num, features)，这里的step_num 是特征的前 (win_size　）　个特征的时间戳特征
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
            opt.zero_grad()
            z, h = model(batched_graphs, batched_feats)
            loss = Loss(h, batched_targets)
            loss.backward()
            opt.step()
            running_loss.append(loss.item())
        epoch_loss = np.mean(running_loss)
        if prev_loss - epoch_loss < early_stop_threshold:
            stop_count += 1
            if stop_count == PATIENCE:
                print('Early stopping')
                model.load_state_dict(best_state_dict)
                break
        else:
            best_state_dict = model.state_dict()
            stop_count = 0
            prev_loss = epoch_loss
        if epoch % 10 == 0:
            print(f'epoch {epoch} loss: ', epoch_loss)
        scheduler.step(np.mean(running_loss))
    return model
