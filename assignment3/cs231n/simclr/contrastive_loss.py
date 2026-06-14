import torch
import numpy as np


def sim(z_i, z_j):
    """两个向量之间的归一化点积。

    输入：
    - z_i: 1xD 张量。
    - z_j: 1xD 张量。
    
    返回：
    - 一个标量值，表示 z_i 和 z_j 之间的归一化点积。
    """
    norm_dot_product = None
    ##############################################################################
    # 你的代码开始处                                                              #
    #                                                                            #
    # 提示：torch.linalg.norm 可能会有帮助。                                      #
    ##############################################################################
    dot = torch.dot(z_i, z_j)
    norm_i = torch.linalg.norm(z_i)
    norm_j = torch.linalg.norm(z_j)
    norm_dot_product = dot / (norm_i * norm_j)
    
    ##############################################################################
    # 你的代码结束处                                                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """计算一个批次上的对比损失 L（朴素循环版本）。
    
    输入：
    - out_left: NxD 张量；投影头 g() 的输出，SimCLR 模型的左分支。
    - out_right: NxD 张量；投影头 g() 的输出，SimCLR 模型的右分支。
    每行是批次中一个增强样本的 z 向量。out_left 和 out_right 中相同行构成一个正样本对。
    换句话说，对于所有 k=0...N-1，(out_left[k], out_right[k]) 构成一个正样本对。
    - tau: 标量值，温度参数，决定指数增长的速度。
    
    返回：
    - 一个标量值；批次中所有正样本对的总损失。定义见笔记本。
    """
    N = out_left.shape[0]  # 训练样本的总数
    
    # 将 out_left 和 out_right 拼接成一个 2*N x D 张量。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # 遍历每个正样本对 (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # 你的代码开始处                                                              #
        #                                                                            #
        # 提示：计算 l(k, k+N) 和 l(k+N, k)。                                         #
        ##############################################################################
        pos_sim1 = sim(z_k, z_k_N)
        numer1 = torch.exp(pos_sim1 / tau)
        
        sum_den1 = 0.0
        for m in range(2 * N):
            if m != k:
                sum_den1 += torch.exp(sim(z_k, out[m]) / tau)
        loss1 = -torch.log(numer1 / sum_den1)
        
        # 2. 计算第二部分：z_k_N 作为锚点，正例 z_k
        pos_sim2 = sim(z_k_N, z_k)
        numer2 = torch.exp(pos_sim2 / tau)
        
        sum_den2 = 0.0
        for m in range(2 * N):
            if m != (k + N):
                sum_den2 += torch.exp(sim(z_k_N, out[m]) / tau)
        loss2 = -torch.log(numer2 / sum_den2)
        
        total_loss += (loss1 + loss2)
        ##############################################################################
        # 你的代码结束处                                                             #
        ##############################################################################
    
    # 最后，我们需要将总损失除以 2N，即批次中的样本数。
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """正样本对之间的归一化点积。

    输入：
    - out_left: NxD 张量；投影头 g() 的输出，SimCLR 模型的左分支。
    - out_right: NxD 张量；投影头 g() 的输出，SimCLR 模型的右分支。
    每行是批次中一个增强样本的 z 向量。
    out_left 和 out_right 中相同行构成一个正样本对。
    
    返回：
    - 一个 Nx1 张量；每行 k 是 out_left[k] 和 out_right[k] 之间的归一化点积。
    """
    pos_pairs = None
    
    ##############################################################################
    # 你的代码开始处                                                              #
    #                                                                            #
    # 提示：torch.linalg.norm 可能会有帮助。                                      #
    ##############################################################################
    
    norm_left = torch.linalg.norm(out_left, dim=1, keepdim=True)  # [N,1]
    norm_right = torch.linalg.norm(out_right, dim=1, keepdim=True)# [N,1]
    dot_product = torch.sum(out_left * out_right, dim=1, keepdim=True)  # [N,1]
    pos_pairs = dot_product / (norm_left * norm_right)
    ##############################################################################
    # 你的代码结束处                                                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """计算批次中所有增强样本对之间归一化点积的 2N x 2N 矩阵。

    输入：
    - out: 2N x D 张量；每行是单个增强样本的 z 向量（投影头的输出）。
    批次中总共有 2N 个增强样本。
    
    返回：
    - sim_matrix: 2N x 2N 张量；矩阵中的每个元素 i, j 是 out[i] 和 out[j] 之间的归一化点积。
    """
    sim_matrix = None
    
    ##############################################################################
    # 你的代码开始处                                                             #
    ##############################################################################
    
    # 向量化计算两两归一化点积矩阵
    norm = torch.linalg.norm(out, dim=1, keepdim=True)  # [2N, 1]
    out_norm = out / norm  # 逐行归一化
    sim_matrix = torch.matmul(out_norm, out_norm.t())  # 矩阵乘法得到两两内积
    
    ##############################################################################
    # 你的代码结束处                                                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """计算批次上的对比损失 L（向量化版本）。不允许使用循环。
    
    输入和输出与 simclr_loss_naive 相同。
    """
    N = out_left.shape[0]
    
    # 将 out_left 和 out_right 拼接成一个 2*N x D 张量。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # 计算批次中所有增强样本对之间的相似度矩阵。
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # 你的代码开始处。按照提示进行。                                             #
    ##############################################################################
    
    # 步骤 1：使用 sim_matrix 计算所有增强样本的分母值。
    # 提示：计算 e^(sim / tau) 并存储到 exponential 中，其形状应为 2N x 2N。
    exponential = torch.exp(sim_matrix / tau)
    
    # 掩码去掉自身匹配项(i==j)
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # 分母：每行求和
    denom = torch.sum(exponential, dim=1, keepdim=True)  # [2N, 1]

    # 步骤 2：提取正样本对相似度
    pos_sim = sim_positive_pairs(out_left, out_right)  # [N, 1]
    # 拼接两组正例：左→右、右→左，拼接后 [2N, 1]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    # 步骤 3：计算分子 exp(pos_sim / tau)
    numerator = torch.exp(pos_sim / tau)  # [2N, 1]

    # 步骤 4：计算单样本损失，再求平均
    loss_per_sample = -torch.log(numerator / denom)
    loss = torch.mean(loss_per_sample)
    
    ##############################################################################
    # 你的代码结束处                                                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    """计算相对误差，用于比较两个值的接近程度。"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))