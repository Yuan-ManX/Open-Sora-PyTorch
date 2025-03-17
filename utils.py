import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


# 定义一个常量，表示每个分块的最大元素数量
NUMEL_LIMIT = 2**30


def ceil_to_divisible(n: int, dividend: int) -> int:
    """
    计算向上取整后的值，使得 dividend 能被 n 整除。

    参数:
    - n (int): 除数。
    - dividend (int): 被除数。

    返回:
    - int: 向上取整后的结果，使得 dividend 能被 n 整除。
    """
    return math.ceil(dividend / (dividend // n))


def chunked_avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    """
    对输入张量进行分块平均池化1D操作，以避免内存不足的问题。

    参数:
    - input (Tensor): 输入张量，形状为 (N, C, L)。
    - kernel_size (int): 池化核的大小。
    - stride (int, 可选): 池化操作的步幅，默认为 kernel_size。
    - padding (int, 可选): 填充大小，默认为0。
    - ceil_mode (bool, 可选): 如果为True，则使用向上取整计算输出大小，默认为False。
    - count_include_pad (bool, 可选): 如果为True，则在计算平均值时包括填充的元素，默认为True。

    返回:
    - Tensor: 平均池化后的输出张量。
    """
    n_chunks = math.ceil(input.numel() / NUMEL_LIMIT)
    if n_chunks == 1:
        # 如果不需要分块，则直接进行平均池化操作
        return F.avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    else:
        # 计算输入和输出的长度
        l_in = input.shape[-1]
        l_out = math.floor((l_in + 2 * padding - kernel_size) / stride + 1)
        output_shape = list(input.shape)
        output_shape[-1] = l_out
        out_list = []

        # 将输入张量按第0维（通常是batch维度）进行分块
        for inp_chunk in input.chunk(n_chunks, dim=0):
            # 对每个分块进行平均池化操作
            out_chunk = F.avg_pool1d(inp_chunk, kernel_size, stride, padding, ceil_mode, count_include_pad)
            out_list.append(out_chunk)
        # 将所有分块的输出沿第0维拼接起来
        return torch.cat(out_list, dim=0)


def chunked_interpolate(input, scale_factor):
    """
    对输入张量进行分块插值操作，以避免内存不足的问题。

    参数:
    - input (Tensor): 输入张量。
    - scale_factor (float): 缩放因子。

    返回:
    - Tensor: 插值后的输出张量。
    """
    output_shape = list(input.shape)
    # 计算输出形状，乘以缩放因子
    output_shape = output_shape[:2] + [int(i * scale_factor) for i in output_shape[2:]]
    n_chunks = math.ceil(torch.Size(output_shape).numel() / NUMEL_LIMIT)
    if n_chunks == 1:
        # 如果不需要分块，则直接进行插值操作
        return F.interpolate(input, scale_factor=scale_factor)
    else:
        out_list = []
        # 增加一个额外的分块以确保覆盖所有数据
        n_chunks += 1
        # 将输入张量按第1维（通常是通道维度）进行分块
        for inp_chunk in input.chunk(n_chunks, dim=1):
            # 对每个分块进行插值操作
            out_chunk = F.interpolate(inp_chunk, scale_factor=scale_factor)
            out_list.append(out_chunk)
        # 将所有分块的输出沿第1维拼接起来
        return torch.cat(out_list, dim=1)


def get_conv3d_output_shape(
    input_shape: torch.Size, out_channels: int, kernel_size: list, stride: list, padding: int, dilation: list
) -> list:
    """
    计算3D卷积的输出形状。

    参数:
    - input_shape (torch.Size): 输入张量的形状。
    - out_channels (int): 输出通道数。
    - kernel_size (List[int]): 卷积核的大小。
    - stride (List[int]): 卷积操作的步幅。
    - padding (int): 填充大小。
    - dilation (List[int]): 膨胀率。

    返回:
    - List[int]: 输出形状列表。
    """
    output_shape = [out_channels]
    if len(input_shape) == 5:
        # 如果输入形状包含batch维度，则在输出形状中保留batch维度
        output_shape.insert(0, input_shape[0])
    # 遍历空间维度（通常是高度、宽度和深度）
    for i, d in enumerate(input_shape[-3:]):
        # 计算输出维度大小
        d_out = math.floor((d + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1)
        output_shape.append(d_out)
    return output_shape


def get_conv3d_n_chunks(numel: int, n_channels: int, numel_limit: int):
    """
    计算3D卷积操作中需要的分块数量。

    参数:
    - numel (int): 输入张量的总元素数量。
    - n_channels (int): 输入通道数。
    - numel_limit (int): 每个分块的最大元素数量。

    返回:
    - int: 分块数量。
    """
    n_chunks = math.ceil(numel / numel_limit)
    n_chunks = ceil_to_divisible(n_chunks, n_channels)
    return n_chunks


def channel_chunk_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list,
    padding: list,
    dilation: list,
    groups: int,
    numel_limit: int,
):
    """
    对3D卷积操作进行分块处理，以避免内存不足的问题。

    参数:
    - input (Tensor): 输入张量。
    - weight (Tensor): 卷积核权重。
    - bias (Tensor): 偏置。
    - stride (List[int]): 卷积操作的步幅。
    - padding (List[int]): 填充大小。
    - dilation (List[int]): 膨胀率。
    - groups (int): 分组数量。
    - numel_limit (int): 每个分块的最大元素数量。

    返回:
    - Tensor: 卷积操作的输出张量。
    """
    out_channels, in_channels = weight.shape[:2]
    kernel_size = weight.shape[2:]
    output_shape = get_conv3d_output_shape(input.shape, out_channels, kernel_size, stride, padding, dilation)
    n_in_chunks = get_conv3d_n_chunks(input.numel(), in_channels, numel_limit)
    n_out_chunks = get_conv3d_n_chunks(
        np.prod(output_shape),
        out_channels,
        numel_limit,
    )
    if n_in_chunks == 1 and n_out_chunks == 1:
        # 如果不需要分块，则直接进行3D卷积操作
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)

    # 如果需要分块，则初始化输出张量
    # output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    # outputs = output.chunk(n_out_chunks, dim=1)
    input_shards = input.chunk(n_in_chunks, dim=1)
    weight_chunks = weight.chunk(n_out_chunks)
    output_list = []
    if bias is not None:
        bias_chunks = bias.chunk(n_out_chunks)
    else:
        bias_chunks = [None] * n_out_chunks
    for weight_, bias_ in zip(weight_chunks, bias_chunks):
        weight_shards = weight_.chunk(n_in_chunks, dim=1)
        o = None
        for x, w in zip(input_shards, weight_shards):
            if o is None:
                o = F.conv3d(x, w, None, stride, padding, dilation, groups).float()
            else:
                o += F.conv3d(x, w, None, stride, padding, dilation, groups).float()
        o = o.to(input.dtype)
        if bias_ is not None:
            o += bias_[None, :, None, None, None]
        # inplace operation cannot be used during training
        # 将结果添加到输出列表中
        # output_.copy_(o)
        output_list.append(o)
    # 将所有分块的输出沿第1维拼接起来
    return torch.cat(output_list, dim=1)


# 定义一个高斯分布类，用于处理对角高斯分布
class DiagonalGaussianDistribution(object):
    """
    对角高斯分布类，用于处理高斯分布的参数、采样和KL散度计算。

    参数:
    - parameters (Tensor): 包含均值和log方差的高斯分布参数，形状为 (N, 2*dim)。
    - deterministic (bool, 可选): 是否为确定性分布。如果是，则方差和标准差设为零。默认为False。
    """
    def __init__(
        self,
        parameters,
        deterministic=False,
    ):
        self.parameters = parameters
        # 将参数分成均值和log方差两部分
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # 对log方差进行裁剪，防止数值不稳定
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        # 计算标准差和方差
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            # 如果是确定性分布，则方差和标准差设为零
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device, dtype=self.mean.dtype)

    def sample(self):
        """
        从高斯分布中采样。

        返回:
        - Tensor: 采样后的张量，形状与均值相同。
        """
        # 从标准正态分布中采样，并调整尺度以匹配当前分布
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device, dtype=self.mean.dtype)
        return x

    def kl(self, other=None):
        """
        计算当前分布与另一个高斯分布之间的KL散度。

        参数:
        - other (DiagonalGaussianDistribution, 可选): 另一个高斯分布。如果为None，则假设另一个分布为标准正态分布。默认为None。

        返回:
        - Tensor: KL散度的标量张量。
        """
        if self.deterministic:
            # 如果是确定性分布，则KL散度为零
            return torch.Tensor([0.0])
        else:
            if other is None: 
                # 如果没有提供另一个分布，则假设另一个分布为标准正态分布
                # 计算KL散度：0.5 * sum(mean^2 + var - 1 - log(var))
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 3, 4]).flatten(0)
            else:
                # 如果提供了另一个分布，
                # 则计算KL散度：0.5 * sum((mean1 - mean2)^2/var2 + var1/var2 - 1 - log(var1) + log(var2))
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 3, 4],
                ).flatten(0)

    def mode(self):
        """
        返回分布的模式（均值）。

        返回:
        - Tensor: 均值张量。
        """
        return self.mean


class ChannelChunkConv3d(nn.Conv3d):
    """
    3D卷积类，支持通道分块以处理大型张量，避免内存不足的问题。

    参数:
    - CONV3D_NUMEL_LIMIT (int): 每个分块的最大元素数量，默认为2^31。
    """
    # 每个分块的最大元素数量
    CONV3D_NUMEL_LIMIT = 2**31

    def _get_output_numel(self, input_shape: torch.Size) -> int:
        """
        计算输出张量的元素数量。

        参数:
        - input_shape (torch.Size): 输入张量的形状。

        返回:
        - int: 输出张量的元素数量。
        """
        numel = self.out_channels
        if len(input_shape) == 5:
            numel *= input_shape[0]
        for i, d in enumerate(input_shape[-3:]):
            d_out = math.floor(
                (d + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) / self.stride[i] + 1
            )
            numel *= d_out
        return numel

    def _get_n_chunks(self, numel: int, n_channels: int):
        """
        计算需要的分块数量。

        参数:
        - numel (int): 输入或输出张量的元素数量。
        - n_channels (int): 输入或输出通道数。

        返回:
        - int: 分块数量。
        """
        n_chunks = math.ceil(numel / ChannelChunkConv3d.CONV3D_NUMEL_LIMIT)
        n_chunks = ceil_to_divisible(n_chunks, n_channels)
        return n_chunks

    def forward(self, input: Tensor) -> Tensor:
        """
        前向传播方法，支持通道分块以处理大型张量。

        参数:
        - input (Tensor): 输入张量。

        返回:
        - Tensor: 输出张量。
        """
        # 如果输入张量的大小小于限制，则直接进行3D卷积操作
        if input.numel() // input.size(0) < ChannelChunkConv3d.CONV3D_NUMEL_LIMIT:
            return super().forward(input)
        # 计算输入和输出需要的分块数量
        n_in_chunks = self._get_n_chunks(input.numel(), self.in_channels)
        n_out_chunks = self._get_n_chunks(self._get_output_numel(input.shape), self.out_channels)
        if n_in_chunks == 1 and n_out_chunks == 1:
            # 如果不需要分块，则直接进行3D卷积操作
            return super().forward(input)
        # 初始化输出列表
        outputs = []
        # 将输入张量按通道维度进行分块
        input_shards = input.chunk(n_in_chunks, dim=1)
        # 对每个输出分块和对应的权重和偏置进行3D卷积操作
        for weight, bias in zip(self.weight.chunk(n_out_chunks), self.bias.chunk(n_out_chunks)):
            weight_shards = weight.chunk(n_in_chunks, dim=1)
            o = None
            for x, w in zip(input_shards, weight_shards):
                if o is None:
                    o = F.conv3d(x, w, bias, self.stride, self.padding, self.dilation, self.groups)
                else:
                    o += F.conv3d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
            outputs.append(o)
        # 将所有输出分块沿通道维度拼接起来
        return torch.cat(outputs, dim=1)


# 使用torch.compile进行编译优化，模式为“最大自动调优且不使用CUDA图”，并启用动态形状
@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def pad_for_conv3d(x: torch.Tensor, width_pad: int, height_pad: int, time_pad: int) -> torch.Tensor:
    """
    对输入张量进行3D卷积所需的填充操作。

    参数:
    - x (Tensor): 输入张量，形状为 (N, C, D, H, W)。
    - width_pad (int): 宽度方向上的填充大小。
    - height_pad (int): 高度方向上的填充大小。
    - time_pad (int): 时间（深度）方向上的填充大小。

    返回:
    - Tensor: 填充后的张量。
    """
    if width_pad > 0 or height_pad > 0:
        # 在宽度和高度方向上进行常数填充，填充值为0
        x = F.pad(x, (width_pad, width_pad, height_pad, height_pad), mode="constant", value=0)
    if time_pad > 0:
        # 在时间（深度）方向上进行复制填充，以复制边界值
        x = F.pad(x, (0, 0, 0, 0, time_pad, time_pad), mode="replicate")
    return x


def pad_for_conv3d_kernel_3x3x3(x: torch.Tensor) -> torch.Tensor:
    """
    对输入张量进行3x3x3卷积核所需的填充操作。

    参数:
    - x (Tensor): 输入张量。

    返回:
    - Tensor: 填充后的张量。
    """
    n_chunks = math.ceil(x.numel() / NUMEL_LIMIT)
    if n_chunks == 1:
        # 如果不需要分块，则进行常数和复制填充
        # 宽度和高度方向上填充1
        x = F.pad(x, (1, 1, 1, 1), mode="constant", value=0)
        # 时间（深度）方向上填充1
        x = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
    else:
        # 如果需要分块，则对每个分块进行填充
        out_list = []
        # 增加一个额外的分块以确保覆盖所有数据
        n_chunks += 1
        for inp_chunk in x.chunk(n_chunks, dim=1):
            out_chunk = F.pad(inp_chunk, (1, 1, 1, 1), mode="constant", value=0)
            out_chunk = F.pad(out_chunk, (0, 0, 0, 0, 1, 1), mode="replicate")
            out_list.append(out_chunk)
        # 将所有填充后的分块沿通道维度拼接起来
        x = torch.cat(out_list, dim=1)
    return x


class PadConv3D(nn.Module):
    """
    3D卷积填充模块，用于在时间（深度）维度上填充第一帧。

    参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - kernel_size (int, 可选): 卷积核大小，默认为3。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        if isinstance(kernel_size, int):
            # 如果kernel_size是整数，则将其转换为3元组，表示3D卷积核的大小
            kernel_size = (kernel_size,) * 3
        self.kernel_size = kernel_size

        # 特定填充
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        assert time_kernel_size == height_kernel_size == width_kernel_size, "only support cubic kernel size"
        if time_kernel_size == 3:
            # 如果卷积核大小为3，则使用特定的填充函数
            self.pad = pad_for_conv3d_kernel_3x3x3
        else:
            assert time_kernel_size == 1, f"only support kernel size 1/3 for now, got {kernel_size}"
            # 如果卷积核大小为1，则无需填充
            self.pad = lambda x: x

        # 定义3D卷积层，填充为0，因为填充已在pad函数中处理
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 卷积后的输出张量。
        """
        # 应用填充
        x = self.pad(x)
        # 进行3D卷积
        x = self.conv(x)
        return x


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
class ChannelChunkPadConv3D(PadConv3D):
    """
    支持通道分块的3D卷积填充模块。

    参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - kernel_size (int, 可选): 卷积核大小，默认为3。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(in_channels, out_channels, kernel_size)
        # 使用支持通道分块的3D卷积层替换默认的Conv3d层
        self.conv = ChannelChunkConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
