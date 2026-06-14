from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """计算仿射（全连接）层的前向传播。

    输入x的形状为(N, d_1, ..., d_k)，包含N个样本的小批量，
    其中每个样本x[i]的形状为(d_1, ..., d_k)。我们会将每个输入重塑为维度为D = d_1 * ... * d_k的向量，
    然后将其转换为维度为M的输出向量。

    输入：
    - x: 包含输入数据的numpy数组，形状为(N, d_1, ..., d_k)
    - w: 权重的numpy数组，形状为(D, M)
    - b: 偏置的numpy数组，形状为(M,)

    返回：
    - out: 输出，形状为(N, M)
    - cache: (x, w, b)，用于反向传播的缓存数据
    """
    out = None
    ###########################################################################
    # 请从作业1中复制你的解决方案。                                             #
    ###########################################################################
    N = x.shape[0]
    # 将高维输入重塑为 (N, D) 的二维矩阵（每个样本展平为一维向量）
    x_reshaped = x.reshape(N, -1)
    # 仿射变换：矩阵乘法 + 偏置  out = XW + b
    out = np.dot(x_reshaped, w) + b
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """计算仿射（全连接）层的反向传播。

    输入：
    - dout: 上游导数，形状为(N, M)
    - cache: 元组，包含：
      - x: 输入数据，形状为(N, d_1, ... d_k)
      - w: 权重，形状为(D, M)
      - b: 偏置，形状为(M,)

    返回：
    - dx: 相对于x的梯度，形状为(N, d1, ..., d_k)
    - dw: 相对于w的梯度，形状为(D, M)
    - db: 相对于b的梯度，形状为(M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # 请从作业1中复制你的解决方案。                                             #
    ###########################################################################
    N = dout.shape[0]
    x_row = x.reshape(N, -1)
    out = x_row.dot(w) + b
    dx_row = dout.dot(w.T)
    dw = x_row.T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dx_row.reshape(x.shape)
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """计算整流线性单元（ReLU）层的前向传播。

    输入：
    - x: 任意形状的输入

    返回：
    - out: 输出，与x形状相同
    - cache: x，用于反向传播的缓存数据
    """
    out = None
    ###########################################################################
    # 请从作业1中复制你的解决方案。                                             #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """计算整流线性单元（ReLU）层的反向传播。

    输入：
    - dout: 上游导数，任意形状
    - cache: 输入x，与dout形状相同

    返回：
    - dx: 相对于x的梯度
    """
    dx, x = None, cache
    ###########################################################################
    # 请从作业1中复制你的解决方案。                                             #
    ###########################################################################
    dx = (x > 0).astype(dout.dtype) * dout
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """计算softmax分类的损失和梯度。

    输入：
    - x: 输入数据，形状为(N, C)，其中x[i, j]是第i个输入在第j类上的得分
    - y: 标签向量，形状为(N,)，其中y[i]是x[i]的标签，且0 <= y[i] < C

    返回：
    - loss: 标量损失值
    - dx: 相对于x的损失梯度
    """
    loss, dx = None, None

    ###########################################################################
    # 请从作业1中复制你的解决方案。                                             #
    ###########################################################################
    N = x.shape[0]

    x_shifted = x - np.max(x, axis=1, keepdims=True)
    
    # 计算概率
    exp_scores = np.exp(x_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 计算损失
    correct_logprobs = -np.log(probs[np.arange(N), y])
    loss = np.sum(correct_logprobs) / N
    
    # 计算梯度
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """批归一化的前向传播。

    在训练期间，样本均值和（未校正的）样本方差从迷你批量统计中计算，并用于归一化输入数据。
    在训练期间，我们还保持每个特征的均值和方差的指数衰减运行平均值，这些平均值用于测试时的归一化。

    在每个时间步，我们使用基于动量参数的指数衰减来更新均值和方差的运行平均值：

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    注意，批归一化论文建议不同的测试时行为：他们使用大量训练图像计算每个特征的样本均值和方差，
    而不是使用运行平均值。在本实现中，我们选择使用运行平均值，因为它们不需要额外的估计步骤；
    torch7的批归一化实现也使用运行平均值。

    输入：
    - x: 形状为(N, D)的数据
    - gamma: 形状为(D,)的缩放参数
    - beta: 形状为(D,)的偏移参数
    - bn_param: 包含以下键的字典：
      - mode: 'train'或'test'；必需
      - eps: 数值稳定性的常数
      - momentum: 运行均值/方差的常数
      - running_mean: 形状为(D,)的特征运行均值数组
      - running_var: 形状为(D,)的特征运行方差数组

    返回：
    - out: 形状为(N, D)的输出
    - cache: 反向传播所需的中间值元组
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        ########################################################################
        # 实现批归一化的训练时前向传播。                                         #
        # 使用迷你批量统计计算均值和方差，使用这些统计量归一化输入数据，           #
        # 并使用gamma和beta对归一化数据进行缩放和偏移。                          #
        #                                                                      #
        # 应将输出存储在变量out中。反向传播所需的任何中间值应存储在cache变量中。   #
        #                                                                      #
        # 还应使用计算出的样本均值和方差以及动量变量来更新运行均值和运行方差，     #
        # 将结果存储在running_mean和running_var变量中。                         #
        #                                                                      #
        # 注意，尽管需要跟踪运行方差，但应基于标准差（方差的平方根）归一化数据！   #
        # 参考原始论文（https://arxiv.org/abs/1502.03167）可能会有帮助。         #
        ########################################################################
        # 计算批量统计量
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        # 归一化
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta
        # 更新运行平均值
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        # 缓存用于反向传播
        cache = (x_hat, x, gamma, sample_mean, sample_var, eps, N)
        ########################################################################
        #                           你的代码结束                                #
        ########################################################################
    elif mode == "test":
        ################################################################################
        # 实现批归一化的测试时前向传播。                                                 #
        # 使用运行均值和方差归一化输入数据，然后使用gamma和beta对归一化数据进行缩放和偏移。#
        # 将结果存储在out变量中。                                                      #
        ##############################################################################
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        #######################################################################
        #                          你的代码结束                                 #
        #######################################################################
    else:
        raise ValueError('无效的批归一化前向模式 "%s"' % mode)

    # 将更新后的运行均值存储回bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """批归一化的反向传播。

    对于本实现，应在纸上画出批归一化的计算图，并通过中间节点反向传播梯度。

    输入：
    - dout: 上游导数，形状为(N, D)
    - cache: 来自batchnorm_forward的中间值变量。

    返回：
    - dx: 相对于输入x的梯度，形状为(N, D)
    - dgamma: 相对于缩放参数gamma的梯度，形状为(D,)
    - dbeta: 相对于偏移参数beta的梯度，形状为(D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # 实现批归一化的反向传播。将结果存储在dx、dgamma和dbeta变量中。              #
    # 参考原始论文（https://arxiv.org/abs/1502.03167）可能会有帮助。            #
    ###########################################################################
    x_hat, x, gamma, sample_mean, sample_var, eps, N = cache

    # 计算 dgamma 和 dbeta
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    # 计算 dx_hat
    dx_hat = dout * gamma

    # 计算标准差的倒数
    std_inv = 1.0 / np.sqrt(sample_var + eps)

    # 计算方差的梯度
    var_grad = np.sum(dx_hat * (x - sample_mean) * (-0.5) * (sample_var + eps) ** (-1.5), axis=0)

    # 计算均值的梯度
    mean_grad = np.sum(dx_hat * (-std_inv), axis=0) + var_grad * np.sum(-2.0 * (x - sample_mean), axis=0) / N

    # 计算关于输入 x 的梯度
    dx = dx_hat * std_inv + var_grad * 2.0 * (x - sample_mean) / N + mean_grad / N
    ###########################################################################
    #                             你的代码结束                                #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """批归一化的替代反向传播。

    对于本实现，应在纸上计算批归一化反向传播的导数并尽可能简化。应能推导出反向传播的简单表达式。
    更多提示参见jupyter笔记本。

    注意：本实现应期望接收与batchnorm_backward相同的cache变量，但可能不会使用cache中的所有值。

    输入/输出：与batchnorm_backward相同
    """
    dx, dgamma, dbeta = None, None, None
    ############################################################################
    # 实现批归一化的反向传播（简化形式）。                                       #
    # 使用推导后的简洁表达式直接计算关于输入的梯度。                             #
    ############################################################################
    x_hat, x, gamma, sample_mean, sample_var, eps, N = cache

    # dgamma 和 dbeta
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    # 中间梯度
    dx_hat = dout * gamma
    std_inv = 1.0 / np.sqrt(sample_var + eps)

    # 简化的 dx 公式
    dx = (1.0 / N) * std_inv * (
        N * dx_hat - np.sum(dx_hat, axis=0) - x_hat * np.sum(dx_hat * x_hat, axis=0)
    )

    ###########################################################################
    #                             你的代码结束                                #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """层归一化的前向传播。

    在训练和测试时，输入数据都按每个数据点进行归一化，然后使用与批归一化相同的gamma和beta参数进行缩放和偏移。

    注意，与批归一化不同，层归一化在训练和测试时的行为是相同的，不需要跟踪任何运行平均值。

    输入：
    - x: 形状为(N, D)的数据
    - gamma: 形状为(D,)的缩放参数
    - beta: 形状为(D,)的偏移参数
    - ln_param: 包含以下键的字典：
        - eps: 数值稳定性的常数

    返回：
    - out: 形状为(N, D)的输出
    - cache: 反向传播所需的中间值元组
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ##############################################################################
    # 实现层归一化的训练时前向传播。                                               #
    # 归一化输入数据，并使用gamma和beta对归一化数据进行缩放和偏移。                 #
    # 提示：这可以通过稍微修改批归一化的训练时实现，并插入一两行精心设计的代码来完成。#
    # 特别是，能否想到任何矩阵变换，可以使你复制批归一化代码并几乎不做修改？         #
    #############################################################################
    # 层归一化：对每个样本的特征维度进行归一化（沿 axis=1）
    # 计算每个样本的均值和方差，保持维度以便后续广播
    sample_mean = np.mean(x, axis=1, keepdims=True)
    sample_var = np.var(x, axis=1, keepdims=True)
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_hat + beta
    # 缓存用于反向传播（与 batchnorm 类似，但样本维度不同）
    cache = (x_hat, x, gamma, sample_mean, sample_var, eps)

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """层归一化的反向传播。

    对于本实现，可以在很大程度上依赖已经为批归一化所做的工作。

    输入：
    - dout: 上游导数，形状为(N, D)
    - cache: 来自layernorm_forward的中间值变量。

    返回：
    - dx: 相对于输入x的梯度，形状为(N, D)
    - dgamma: 相对于缩放参数gamma的梯度，形状为(D,)
    - dbeta: 相对于偏移参数beta的梯度，形状为(D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # 实现层归一化的反向传播。                                                  #
    # 对每个样本（沿特征维度）求导，思路与 batchnorm_backward 类似，但求和轴不同。 #
    ###########################################################################
    x_hat, x, gamma, sample_mean, sample_var, eps = cache

    N, D = x.shape

    # dgamma 和 dbeta（与 batchnorm 一致，为每个特征累加所有样本的贡献）
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    # 中间梯度
    dx_hat = dout * gamma
    std_inv = 1.0 / np.sqrt(sample_var + eps)  # 形状为 (N,1)

    # 对每个样本在特征维度上求和（axis=1，保持维度以便广播）
    var_grad = np.sum(dx_hat * (x - sample_mean) * (-0.5) * (sample_var + eps) ** (-1.5), axis=1, keepdims=True)

    mean_grad = np.sum(dx_hat * (-std_inv), axis=1, keepdims=True) + var_grad * np.sum(-2.0 * (x - sample_mean), axis=1, keepdims=True) / D

    dx = dx_hat * std_inv + var_grad * 2.0 * (x - sample_mean) / D + mean_grad / D

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """倒置丢弃法（inverted dropout）的前向传播。

    注意这与标准丢弃法不同。这里，p是保留神经元输出的概率，而非丢弃神经元输出的概率。
    更多细节参见http://cs231n.github.io/neural-networks-2/#reg。

    输入：
    - x: 任意形状的输入数据
    - dropout_param: 包含以下键的字典：
      - p: 丢弃参数。我们以概率p保留每个神经元的输出。
      - mode: 'test'或'train'。若为训练模式，则执行丢弃；若为测试模式，则直接返回输入。
      - seed: 随机数生成器的种子。传入种子可使函数具有确定性，这在梯度检查中需要，但在实际网络中不需要。

    输出：
    - out: 与x形状相同的数组。
    - cache: 元组(dropout_param, mask)。训练模式下，mask是用于与输入相乘的丢弃掩码；测试模式下，mask为None。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])  # 设置随机种子以保证确定性

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # 实现训练阶段的倒置丢弃法前向传播。将丢弃掩码存储在mask变量中。         #
        #######################################################################
        # 生成与输入同形状的掩码，保留概率为p
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           你的代码结束                               #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # 实现测试阶段的倒置丢弃法前向传播。                                    #
        #######################################################################
        # 测试模式下不进行dropout，直接返回输入
        out = x
        #######################################################################
        #                            你的代码结束                              #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)  # 确保输出数据类型与输入一致

    return out, cache


def dropout_backward(dout, cache):
    """倒置丢弃法的反向传播。

    输入：
    - dout: 上游导数，任意形状
    - cache: 来自dropout_forward的(dropout_param, mask)
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # 实现训练阶段的倒置丢弃法反向传播。                                    #
        #######################################################################
        # 训练时只需将上游梯度乘以前向保存的mask
        dx = dout * mask
        #######################################################################
        #                          你的代码结束                                 #
        #######################################################################
    elif mode == "test":
        dx = dout  # 测试模式下，梯度直接传递
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """卷积层前向传播的朴素实现。

    输入包含N个数据点，每个数据点有C个通道、高度H和宽度W。我们使用F个不同的滤波器对每个输入进行卷积，
    每个滤波器覆盖所有C个通道，高度为HH，宽度为WW。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - w: 滤波器权重，形状为(F, C, HH, WW)
    - b: 偏置，形状为(F,)
    - conv_param: 包含以下键的字典：
      - 'stride': 水平和垂直方向上相邻感受野之间的像素数（步长）。
      - 'pad': 用于对输入进行零填充的像素数。

    填充时，应在输入的高度和宽度轴上对称地放置'pad'个零（即两侧各放pad个）。注意不要直接修改原始输入x。

    返回：
    - out: 输出数据，形状为(N, F, H', W')，其中H'和W'由下式计算：
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # 实现卷积前向传播。提示：可以使用np.pad函数进行填充。                       #
    ###########################################################################
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # 输出尺寸
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # 填充输入
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    window = x_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]
    
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """卷积层反向传播的朴素实现。

    输入：
    - dout: 上游导数。
    - cache: 来自conv_forward_naive的(x, w, b, conv_param)元组

    返回：
    - dx: 相对于x的梯度
    - dw: 相对于w的梯度
    - db: 相对于b的梯度
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # 实现卷积反向传播。                                                       #
    ###########################################################################
    x, w, b, conv_param = cache
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # 输出尺寸
    _, _, H_out, W_out = dout.shape

    # 初始化梯度
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # 对输入进行填充，便于计算dx
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.zeros_like(x_padded)

    # db: 对每个滤波器在所有样本与空间位置上求和
    db = np.sum(dout, axis=(0, 2, 3))

    # 对于每个输出位置，累加 dw 和 dx_padded
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    window = x_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]
                    dout_val = dout[n, f, i, j]
                    dw[f] += dout_val * window
                    dx_padded[n, :, h_start:h_start + HH, w_start:w_start + WW] += dout_val * w[f]

    # 去掉填充得到 dx
    if pad == 0:
        dx = dx_padded
    else:
        dx = dx_padded[:, :, pad:-pad, pad:-pad]

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """最大池化层前向传播的朴素实现。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - pool_param: 包含以下键的字典：
      - 'pool_height': 每个池化区域的高度
      - 'pool_width': 每个池化区域的宽度
      - 'stride': 相邻池化区域之间的距离

    这里不需要填充，例如可假设：
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    返回：
    - out: 输出数据，形状为(N, C, H', W')，其中H'和W'由下式计算：
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # 实现最大池化前向传播。                                                   #
    ###########################################################################
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')

    N, C, H, W = x.shape

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    window = x[n, c, h_start:h_start + pool_height, w_start:w_start + pool_width]
                    out[n, c, i, j] = np.max(window)
    
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """最大池化层反向传播的朴素实现。

    输入：
    - dout: 上游导数
    - cache: 来自前向传播的(x, pool_param)元组

    返回：
    - dx: 相对于x的梯度
    """
    dx = None
    ###########################################################################
    # 实现最大池化反向传播。                                                   #
    ###########################################################################
    x, pool_param = cache
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')

    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    window = x[n, c, h_start:h_start + pool_height, w_start:w_start + pool_width]
                    max_val = np.max(window)
                    mask = (window == max_val)
                    dx[n, c, h_start:h_start + pool_height, w_start:w_start + pool_width] += dout[n, c, i, j] * mask

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """空间批归一化的前向传播。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - gamma: 缩放参数，形状为(C,)
    - beta: 偏移参数，形状为(C,)
    - bn_param: 包含以下键的字典：
      - mode: 'train'或'test'；必需
      - eps: 数值稳定性常数
      - momentum: 运行均值/方差的常数。momentum=0表示每次完全丢弃旧信息，
        而momentum=1表示从不纳入新信息。默认momentum=0.9在大多数情况下适用。
      - running_mean: 形状为(D,)的特征运行均值数组
      - running_var: 形状为(D,)的特征运行方差数组

    返回：
    - out: 输出数据，形状为(N, C, H, W)
    - cache: 反向传播所需的值
    """
    out, cache = None, None

    ###########################################################################
    # 实现空间批归一化的前向传播。                                              #
    #                                                                         #
    # 提示：可通过调用上面实现的标准批归一化函数来实现空间批归一化。              #
    # 实现应该非常简短；我们的实现不到5行。                                     #
    ###########################################################################
    # 
    N, C, H, W = x.shape
    # 将 (N, C, H, W) -> (N*H*W, C) 以便使用 batchnorm_forward
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """空间批归一化的反向传播。

    输入：
    - dout: 上游导数，形状为(N, C, H, W)
    - cache: 前向传播中的值

    返回：
    - dx: 相对于输入的梯度，形状为(N, C, H, W)
    - dgamma: 相对于缩放参数的梯度，形状为(C,)
    - dbeta: 相对于偏移参数的梯度，形状为(C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # 实现空间批归一化的反向传播。                                              #
    #                                                                         #
    # 提示：可通过调用上面实现的标准批归一化函数来实现空间批归一化。              #
    # 实现应该非常简短；我们的实现不到5行。                                     #
    ###########################################################################
    N, C, H, W = dout.shape
    # 将 (N, C, H, W) -> (N*H*W, C)
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """空间组归一化的前向传播。
    
    与层归一化不同，组归一化将数据中的每个样本分成G个连续的部分，然后独立地对每个部分进行归一化。
    然后对数据应用逐特征的偏移和缩放，方式与批归一化和层归一化相同。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - gamma: 缩放参数，形状为(1, C, 1, 1)
    - beta: 偏移参数，形状为(1, C, 1, 1)
    - G: 要划分的组数，必须是C的约数
    - gn_param: 包含以下键的字典：
      - eps: 数值稳定性常数

    返回：
    - out: 输出数据，形状为(N, C, H, W)
    - cache: 反向传播所需的值
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # 实现空间组归一化的前向传播。                                              #
    # 这与层归一化的实现极其相似。                                              #
    # 具体来说，思考如何转换矩阵，使得大部分代码可复用训练时的批归一化和层归一化！ #
    ###########################################################################
    N, C, H, W = x.shape

    # 重塑为 (N, G, C//G, H, W) 以便对每个组计算均值和方差
    x_grouped = x.reshape(N, G, C // G, H, W)

    # 计算组内均值和方差（对通道子维和空间维求均值）
    mean = np.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_grouped, axis=(2, 3, 4), keepdims=True)

    # 归一化
    x_groupnorm = (x_grouped - mean) / np.sqrt(var + eps)

    # 恢复原始形状 (N, C, H, W)
    x_hat = x_groupnorm.reshape(N, C, H, W)

    # 应用可学习的缩放和平移参数（gamma, beta形状应为(1,C,1,1)或(C,)）
    out = gamma * x_hat + beta

    cache = (G, x, x_hat, mean, var, eps, gamma, beta)

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """空间组归一化的反向传播。

    输入：
    - dout: 上游导数，形状为(N, C, H, W)
    - cache: 前向传播中的值

    返回：
    - dx: 相对于输入的梯度，形状为(N, C, H, W)
    - dgamma: 相对于缩放参数的梯度，形状为(1, C, 1, 1)
    - dbeta: 相对于偏移参数的梯度，形状为(1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # 实现空间组归一化的反向传播。                                              #
    # 这与层归一化的实现极其相似。                                              #
    ###########################################################################
    G, x, x_hat, mean, var, eps, gamma, beta = cache

    N, C, H, W = dout.shape
    # dgamma 和 dbeta：对 N,H,W 求和，保留通道维度
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    # 计算 dx
    # 先计算 dx_hat = dout * gamma
    dx_hat = dout * gamma

    # 重塑为组结构 (N, G, C//G, H, W)
    Cg = C // G
    x_hat_group = x_hat.reshape(N, G, Cg, H, W)
    dx_hat_group = dx_hat.reshape(N, G, Cg, H, W)

    # 将每组展平为 (N*G, m) 以便使用简化公式，其中 m = Cg*H*W
    m = Cg * H * W
    dx_hat_flat = dx_hat_group.reshape(N * G, m)
    x_hat_flat = x_hat_group.reshape(N * G, m)

    # 标准差的倒数
    std_inv = 1.0 / np.sqrt(var + eps)
    std_inv_flat = std_inv.reshape(N * G, 1)

    # 使用层归一化的简化反向公式
    dx_flat = (1.0 / m) * std_inv_flat * (
        m * dx_hat_flat
        - np.sum(dx_hat_flat, axis=1, keepdims=True)
        - x_hat_flat * np.sum(dx_hat_flat * x_hat_flat, axis=1, keepdims=True)
    )

    # 恢复组形状并再恢复回 (N, C, H, W)
    dx = dx_flat.reshape(N, G, Cg, H, W).reshape(N, C, H, W)

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return dx, dgamma, dbeta
