import os
import torch.nn as nn

from registry import MODELS
from utils.ckpt import load_checkpoint


def weights_init(m):
    """
    权重初始化函数，适用于包含Conv层的模块。

    参数:
    - m (nn.Module): 要初始化的模型或模块。
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # 对于Conv层，使用正态分布初始化权重，均值为0，标准差为0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        # 对于BatchNorm层，使用正态分布初始化权重，均值为1，标准差为0.02
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # 将偏置初始化为常数0
        nn.init.constant_(m.bias.data, 0)


def weights_init_conv(m):
    """
    权重初始化函数，适用于包含conv属性的模块。

    参数:
    - m (nn.Module): 要初始化的模型或模块。
    """
    if hasattr(m, "conv"):
        # 如果模块有conv属性，则将m指向conv属性
        m = m.conv
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # 对于Conv层，使用正态分布初始化权重，均值为0，标准差为0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        # 对于BatchNorm层，使用正态分布初始化权重，均值为1，标准差为0.02
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # 将偏置初始化为常数0
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator3D(nn.Module):
    """
    定义一个3D PatchGAN判别器，类似于Pix2Pix中的判别器，但用于3D输入。

    参数:
    - input_nc (int): 输入体积的通道数，默认为1。
    - ndf (int): 最后一层卷积层的滤波器数量，默认为64。
    - n_layers (int): 判别器中卷积层的数量，默认为5。
    - norm_layer (nn.Module): 归一化层类型，默认为nn.BatchNorm3d。
    - conv_cls (str): 卷积层的类型，默认为"conv3d"。
    - dropout (float): Dropout的概率，默认为0.30。
    """

    def __init__(
        self,
        input_nc=1,
        ndf=64,
        n_layers=5,
        norm_layer=nn.BatchNorm3d,
        conv_cls="conv3d",
        dropout=0.30,
    ):
        super(NLayerDiscriminator3D, self).__init__()
        assert conv_cls == "conv3d"

        # 是否使用偏置
        use_bias = False

        # 卷积核大小
        kw = 3
        # 卷积填充大小
        padw = 1

        # 构建判别器的主干序列
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        # 当前滤波器的倍数
        nf_mult = 1
        # 前一层的滤波器倍数
        nf_mult_prev = 1
        for n in range(1, n_layers):  
            # 逐步增加滤波器的数量
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=2 if n == 1 else (1, 2, 2),
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(dropout),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=(kw, kw, kw),
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        """
        标准的前向传播方法。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 输出张量。
        """
        return self.main(x)


# 注册模块的函数，用于注册3D判别器模型
@MODELS.register_module("N_Layer_discriminator_3D")
def N_LAYER_DISCRIMINATOR_3D(from_pretrained=None, force_huggingface=None, **kwargs):
    """
    创建并初始化3D PatchGAN判别器模型。

    参数:
    - from_pretrained (str, 可选): 预训练模型的路径或标识符。
    - force_huggingface (bool, 可选): 是否强制从HuggingFace加载预训练模型。
    - **kwargs: 其他关键字参数，用于传递给NLayerDiscriminator3D构造函数。

    返回:
    - NLayerDiscriminator3D: 初始化并加载了预训练参数的3D判别器模型。
    """
    # 创建模型实例并应用权重初始化函数
    model = NLayerDiscriminator3D(**kwargs).apply(weights_init)
    
    if from_pretrained is not None:
        if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
            raise NotImplementedError
        else:
            # 加载预训练模型检查点
            load_checkpoint(model, from_pretrained)
        print(f"loaded model from: {from_pretrained}")

    return model
