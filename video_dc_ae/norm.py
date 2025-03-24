from typing import Optional
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from vo_ops import build_kwargs_from_config


# 定义一个列表，包含所有公开的接口名称
__all__ = ["LayerNorm2d", "build_norm", "set_norm_eps"]


class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm2d 类实现了针对二维输入（批量，通道，高度，宽度）的层归一化。

    继承自:
        torch.nn.LayerNorm
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，实现层归一化。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 计算均值，沿通道维度计算每个通道的均值
        out = x - torch.mean(x, dim=1, keepdim=True)
        # 计算标准差，沿通道维度计算每个通道的标准差
        # 计算归一化后的输出
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            # 如果启用了元素级仿射变换，则应用权重和偏置
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        # 返回归一化后的张量
        return out


class RMSNorm2d(nn.Module):
    """
    RMSNorm2d 类实现了针对二维输入（批量，通道，高度，宽度）的 RMS 归一化。

    参数:
        num_features (int): 输入的特征数（通道数）。
        eps (float, optional): 分母中的一个小常数，避免除以零。默认为 1e-5。
        elementwise_affine (bool, optional): 是否应用元素级仿射变换（权重和偏置）。默认为 True。
        bias (bool, optional): 是否包含偏置参数。默认为 True。
    """
    def __init__(
        self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True
    ) -> None:
        super().__init__()
        # 特征数（通道数）
        self.num_features = num_features
        # 小常数
        self.eps = eps
        # 是否应用元素级仿射变换
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            # 初始化权重参数
            self.weight = torch.nn.parameter.Parameter(torch.empty(self.num_features))
            if bias:
                # 初始化偏置参数（如果启用）
                self.bias = torch.nn.parameter.Parameter(torch.empty(self.num_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，实现 RMS 归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 计算 RMS 值，并进行归一化
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        # 如果启用了元素级仿射变换，则应用权重和偏置
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        # 返回归一化后的张量
        return x


class RMSNorm3d(RMSNorm2d):
    """
    RMSNorm3d 类实现了针对三维输入（批量，通道，时间，高度，宽度）的 RMS 归一化。

    继承自:
        RMSNorm2d
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，实现三维 RMS 归一化。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, C, T, H, W)。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 计算 RMS 值，并进行归一化
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        
        if self.elementwise_affine:
            # 如果启用了元素级仿射变换，则应用权重和偏置
            x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        # 返回归一化后的张量
        return x


# 注册归一化函数，键为归一化名称，值为对应的类
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "rms2d": RMSNorm2d,
    "rms3d": RMSNorm3d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    """
    根据名称和参数构建归一化层。

    参数:
        name (str, optional): 归一化层的名称，必须在 REGISTERED_NORM_DICT 中注册。默认为 "bn2d"。
        num_features (int, optional): 输入的特征数（通道数）。默认为 None。
        **kwargs: 其他传递给归一化层类的关键字参数。

    返回:
        Optional[nn.Module]: 如果归一化名称有效，则返回对应的归一化层；否则返回 None。
    """
    if name in ["ln", "ln2d"]:
        # 设置归一化层的标准化形状
        kwargs["normalized_shape"] = num_features
    else:
        # 设置归一化层的特征数
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        # 获取归一化层的类
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    """
    设置归一化层的小常数 eps。

    参数:
        model (nn.Module): 要修改的模型。
        eps (Optional[float], optional): 要设置的小常数。默认为 None。
    """
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                # 如果提供了 eps，则设置归一化层的小常数
                m.eps = eps
