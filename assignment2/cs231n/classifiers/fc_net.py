from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """多层全连接神经网络类。

    该网络包含任意数量的隐藏层、ReLU非线性激活函数和softmax损失函数。
    它还可以选择实现dropout（丢弃法）和batch/layer normalization（批归一化/层归一化）。
    对于一个有L层的网络，其架构为：

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    其中批归一化/层归一化和dropout是可选的，且{...}块会重复(L - 1)次。

    可学习参数存储在self.params字典中，将通过Solver类进行学习。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """初始化一个新的全连接网络。

        输入：
        - hidden_dims: 整数列表，给出每个隐藏层的大小。
        - input_dim: 整数，给出输入的大小。
        - num_classes: 整数，给出要分类的类别数量。
        - dropout_keep_ratio: 0到1之间的标量，给出dropout的强度。
          如果dropout_keep_ratio=1，则网络不使用dropout。
        - normalization: 网络应使用的归一化类型。有效值为"batchnorm"、"layernorm"，
          或None（默认值，不使用归一化）。
        - reg: 标量，给出L2正则化强度。
        - weight_scale: 标量，给出权重随机初始化的标准差。
        - dtype: numpy数据类型对象；所有计算将使用此数据类型。float32更快但精度较低，
          因此数值梯度检查应使用float64。
        - seed: 如果不为None，则将此随机种子传递给dropout层。这将使dropout层具有确定性，
          以便我们可以对模型进行梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1  # 是否使用dropout
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)  # 总层数 = 1（输出层） + 隐藏层数量
        self.dtype = dtype
        self.params = {}  # 存储网络参数的字典

        ##############################################################################
        # 代办: 初始化网络的参数，将所有值存储在 self.params 字典中。将第一层的权重和偏置#
        # 存储在 W1 和 b1 中；第二层存储在 W2 和 b2 中，依此类推。权重应从均值为 0 的正态#
        # 分布中初始化，标准差等于 weight_scale。偏置应初始化为零。                     #
        #                                                                            #
        # 当使用批量归一化时，应将比例和偏移参数存储在第一层的 gamma1 和 beta1 中；第二  #
        # 层存储在 gamma2 和 beta2 中，依此类推。比例参数应初始化为一，偏移参数应初始化  #
        # 为零。                                                                     #
        ##############################################################################

        # 计算网络总层数（包括输入层到第一隐藏层，隐藏层之间，以及最后一层到输出层）
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        
        # 初始化所有层的权重和偏置
        for i in range(self.num_layers):
            W_key = f'W{i+1}'
            b_key = f'b{i+1}'
            # 权重：从均值为0、标准差为weight_scale的高斯分布初始化
            self.params[W_key] = np.random.randn(layer_dims[i], layer_dims[i+1]) * weight_scale
            # 偏置：初始化为0
            self.params[b_key] = np.zeros(layer_dims[i+1])
            
            # 如果使用批量归一化，初始化gamma和beta参数
            if self.normalization == "batchnorm" and i < self.num_layers - 1:
                self.params[f'gamma{i+1}'] = np.ones(layer_dims[i+1])
                self.params[f'beta{i+1}'] = np.zeros(layer_dims[i+1])
            # 如果使用层归一化，同样初始化 gamma/beta（与 batchnorm 相同的形状）
            if self.normalization == "layernorm" and i < self.num_layers - 1:
                self.params[f'gamma{i+1}'] = np.ones(layer_dims[i+1])
                self.params[f'beta{i+1}'] = np.zeros(layer_dims[i+1])

        ############################################################################
        #                             你的代码结束                                  #
        ############################################################################

        # 当使用dropout时，我们需要向每个dropout层传递一个dropout_param字典，
        # 以便该层知道dropout概率和模式（训练/测试）。可以向每个dropout层传递相同的dropout_param。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed  # 设置随机种子以保证确定性

        # 使用批归一化时，我们需要跟踪运行均值和方差，
        # 因此需要向每个批归一化层传递一个特殊的bn_param对象。
        # 应将self.bn_params[0]传递给第一个批归一化层的前向传播，
        # self.bn_params[1]传递给第二个批归一化层的前向传播，依此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # 将所有参数转换为正确的数据类型
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """计算全连接网络的损失和梯度。
        
        输入：
        - X: 输入数据数组，形状为(N, d_1, ..., d_k)
        - y: 标签数组，形状为(N,)。y[i]给出X[i]的标签。

        返回：
        如果y为None，则运行模型的测试时前向传播并返回：
        - scores: 形状为(N, C)的数组，给出分类得分，其中scores[i, c]是X[i]对类别c的分类得分。

        如果y不为None，则运行训练时的前向和反向传播，并返回一个元组：
        - loss: 标量损失值
        - grads: 与self.params具有相同键的字典，将参数名称映射到损失相对于这些参数的梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"  # 根据y是否为None判断是测试还是训练模式

        # 为批归一化参数和dropout参数设置训练/测试模式，因为它们在训练和测试时的行为不同
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ###############################################################################
        # 代办: 实现全连接网络的前向传播，计算 X 的类别分数，并将它们存储在 scores 变量中。#
        #                                                                              #
        # 当使用 Dropout 时，你需要将 self.dropout_param 传递给每个 Dropout 前向传播。    #
        #                                                                              # 
        # 当使用批量归一化时，你需要将 self.bn_params[0] 传递给第一个批量归一化层的前向    #
        # 传播，self.bn_params[1] 传递给第二个批量归一化层的前向传播，依此类推。          #
        ###############################################################################
        
        # 存储所有层的cache，用于反向传播
        caches = []
        
        # 前向传播：处理所有隐藏层（第1层到第num_layers-1层）
        out = X
        for i in range(self.num_layers - 1):
            W_key = f'W{i+1}'
            b_key = f'b{i+1}'
            
            # 仿射层
            affine_out, affine_cache = affine_forward(out, self.params[W_key], self.params[b_key])
            
            cache_layer = []
            
            # 归一化（若需）：支持 batchnorm 与 layernorm
            if self.normalization == "batchnorm":
                gamma_key = f'gamma{i+1}'
                beta_key = f'beta{i+1}'
                bn_out, bn_cache = batchnorm_forward(affine_out, self.params[gamma_key], self.params[beta_key], self.bn_params[i])
                # ReLU激活
                out, relu_cache = relu_forward(bn_out)
                cache_layer = [affine_cache, bn_cache, relu_cache]
            elif self.normalization == "layernorm":
                gamma_key = f'gamma{i+1}'
                beta_key = f'beta{i+1}'
                ln_out, ln_cache = layernorm_forward(affine_out, self.params[gamma_key], self.params[beta_key], self.bn_params[i])
                out, relu_cache = relu_forward(ln_out)
                cache_layer = [affine_cache, ln_cache, relu_cache]
            else:
                # ReLU激活
                out, relu_cache = relu_forward(affine_out)
                cache_layer = [affine_cache, relu_cache]
            
            # Dropout（如果需要）
            if self.use_dropout:
                out, dropout_cache = dropout_forward(out, self.dropout_param)
                cache_layer.append(dropout_cache)
            
            caches.append(cache_layer)
        
        # 最后一层：仅仿射层（Softmax损失在后面计算）
        W_last = f'W{self.num_layers}'
        b_last = f'b{self.num_layers}'
        scores, last_cache = affine_forward(out, self.params[W_last], self.params[b_last])
        caches.append(last_cache)

        ############################################################################
        #                             你的代码结束                                  #
        ############################################################################

        # 如果是测试模式，提前返回
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ##################################################################################
        # 代办: 实现全连接网络的反向传播。将损失存储在 loss 变量中，梯度存储在 grads 字典中。#
        # 计算数据损失时使用 Softmax，并确保 grads[k] 持有相对于 self.params[k] 的损失梯度。#
        # 不要忘记添加 L2 正则化！                                                        #
        #                                                                                #
        # 当使用批量 / 层归一化时，你不需要正则化比例和偏移参数。                           #
        #                                                                               #
        # 注意：为了确保你的实现与我们的匹配，并且你通过了自动化测试，确保你的 L2 正则化     #
        # 包含一个 0.5 的因子，以简化梯度表达式。                                         #
        #################################################################################
        
        N = X.shape[0]
        
        # 计算Softmax损失
        loss, dout = softmax_loss(scores, y)
        
        # 添加L2正则化损失
        for i in range(self.num_layers):
            W_key = f'W{i+1}'
            loss += 0.5 * self.reg * np.sum(self.params[W_key] ** 2)
        
        # 反向传播：先从最后一层仿射层开始
        dout, dw, db = affine_backward(dout, caches[-1])
        grads[f'W{self.num_layers}'] = dw + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db
        
        # 反向传播隐藏层（从倒数第二层开始往前）
        for i in range(self.num_layers - 2, -1, -1):
            cache = caches[i]
            
            # 处理dropout（如果存在）
            if self.use_dropout:
                dout = dropout_backward(dout, cache[-1])
            
            # 处理relu (在dropout之后，总是倒数第二个或第三个)
            if self.normalization == "batchnorm" or self.normalization == "layernorm":
                relu_idx = -2 if self.use_dropout else -1
                dout = relu_backward(dout, cache[relu_idx])

                # 处理 batchnorm 或 layernorm
                bn_idx = -3 if self.use_dropout else -2
                if self.normalization == "batchnorm":
                    dout, dgamma, dbeta = batchnorm_backward(dout, cache[bn_idx])
                else:
                    dout, dgamma, dbeta = layernorm_backward(dout, cache[bn_idx])
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta

                # 处理affine
                affine_idx = -4 if self.use_dropout else -3
                dout, dw, db = affine_backward(dout, cache[affine_idx])
            else:
                # 无batchnorm：cache = [affine_cache, relu_cache, ...]
                relu_idx = -2 if self.use_dropout else -1
                dout = relu_backward(dout, cache[relu_idx])
                affine_idx = -3 if self.use_dropout else -2
                dout, dw, db = affine_backward(dout, cache[affine_idx])
            
            grads[f'W{i+1}'] = dw + self.reg * self.params[f'W{i+1}']
            grads[f'b{i+1}'] = db

        ############################################################################
        #                             你的代码结束                                  #
        ############################################################################

        return loss, grads