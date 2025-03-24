import math
from inspect import signature
from typing import Any, Callable, Optional, Union
import torch
import torch.nn.functional as F


# 定义一个全局变量，用于控制是否打印调试信息
VERBOSE = False


def pixel_shuffle_3d(x, upscale_factor):
    """
    三维像素重排（Pixel Shuffle）操作，用于上采样。

    参数:
        x (torch.Tensor): 输入张量，形状为 (B, C, T, H, W)。
        upscale_factor (int): 上采样因子，用于扩大空间维度。

    返回:
        torch.Tensor: 上采样后的张量，形状为 (B, C // (r^3), T * r, H * r, W * r)。
                     其中 r 是上采样因子。

    异常:
        AssertionError: 如果通道数不是上采样因子的立方倍数，则抛出异常。
    """
    # 解包输入张量的维度
    B, C, T, H, W = x.shape
    # 获取上采样因子
    r = upscale_factor
    assert C % (r * r * r) == 0, "通道数必须是上采样因子的立方倍数"
    
    # 计算新的通道数
    C_new = C // (r * r * r)
    # 重塑张量形状，以分离空间维度
    x = x.view(B, C_new, r, r, r, T, H, W)
    if VERBOSE:
        print("x.view:")
        print(x)
        print("x.view.shape:")
        print(x.shape)
    
    # 重新排列维度顺序
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    if VERBOSE:
        print("x.permute:")
        print(x)
        print("x.permute.shape:")
        print(x.shape)

    # 重塑张量形状，实现上采样
    y = x.reshape(B, C_new, T * r, H * r, W * r)

    # 返回上采样后的张量
    return y


def pixel_unshuffle_3d(x, downsample_factor):
    """
    三维像素逆重排（Pixel Unshuffle）操作，用于下采样。

    参数:
        x (torch.Tensor): 输入张量，形状为 (B, C, T, H, W)。
        downsample_factor (int): 下采样因子，用于缩小空间维度。

    返回:
        torch.Tensor: 下采样后的张量，形状为 (B, C * (r^3), T // r, H // r, W // r)。
                     其中 r 是下采样因子。

    异常:
        AssertionError: 如果时间、高度或宽度维度不是下采样因子的倍数，则抛出异常。
    """
    # 解包输入张量的维度
    B, C, T, H, W = x.shape

    # 获取下采样因子
    r = downsample_factor
    assert T % r == 0, f"时间维度必须是下采样因子的倍数, got shape {x.shape}"
    assert H % r == 0, f"高度维度必须是下采样因子的倍数, got shape {x.shape}"
    assert W % r == 0, f"宽度维度必须是下采样因子的倍数, got shape {x.shape}"

    # 计算新的时间维度大小
    T_new = T // r
    # 计算新的高度维度大小
    H_new = H // r
    # 计算新的宽度维度大小
    W_new = W // r
    # 计算新的通道数
    C_new = C * (r * r * r)
    
    # 重塑张量形状，以分离空间维度
    x = x.view(B, C, T_new, r, H_new, r, W_new, r)
    # 重新排列维度顺序
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    # 重塑张量形状，实现下采样
    y = x.reshape(B, C_new, T_new, H_new, W_new)

    # 返回下采样后的张量
    return y


def test_pixel_shuffle_3d():
    """
    测试函数，用于验证 pixel_shuffle_3d 和 pixel_unshuffle_3d 的正确性。
    """
    # 输入张量 (B, C, T, H, W) = (1, 16, 2, 4, 4)
    # 创建一个示例输入张量
    x = torch.arange(1, 1 + 1 * 16 * 2 * 4 * 4).view(1, 16, 2, 4, 4).float()
    print("x:")
    print(x)
    print("x.shape:")
    print(x.shape)
    
    # 设置上采样因子
    upscale_factor = 2

    # 使用自定义 pixel_shuffle_3d 进行上采样
    y = pixel_shuffle_3d(x, upscale_factor)
    print("pixelshuffle_3d 结果:")
    print(y)
    print("输出形状:", y.shape)
    # 预期输出形状: (1, 1, 4, 8, 8)
    # 因为:
    # - 通道数从16变为1 (16 / (2^3))
    # - 时间维度从2变为4 (2 * 2)
    # - 高度从4变为8 (4 * 2)
    # - 宽度从4变为8 (4 * 2)

    # 验证 pixel_shuffle_3d 和 pixel_unshuffle_3d 的可逆性
    print(torch.allclose(x, pixel_unshuffle_3d(y, upscale_factor)))


def chunked_interpolate(x, scale_factor, mode="nearest"):
    """
    通过在通道维度上分块，对大尺寸张量进行插值操作。
    仅支持 'nearest' 插值模式。

    参数:
        x (torch.Tensor): 输入张量，形状为 (B, C, D, H, W)。
        scale_factor (Tuple[float, float, float]): 缩放因子，元组形式 (d, h, w)。
        mode (str, optional): 插值模式。默认为 "nearest"。目前仅支持 "nearest"。

    返回:
        torch.Tensor: 插值后的张量。

    异常:
        AssertionError: 如果插值模式不是 "nearest"，则抛出异常。
        ValueError: 如果输入张量不是 5 维，则抛出异常。
    """
    assert (
        mode == "nearest"
    ), "Only the nearest mode is supported"  # actually other modes are theoretically supported but not tested
    if len(x.shape) != 5:
        raise ValueError("Expected 5D input tensor (B, C, D, H, W)")

    # 计算避免 int32 溢出的最大分块元素数。num_elements < max_int32
    # 最大 int32 是 2^31 - 1
    max_elements_per_chunk = 2**31 - 1

    # 计算输出空间维度
    # 计算输出深度
    out_d = math.ceil(x.shape[2] * scale_factor[0])
    # 计算输出高度
    out_h = math.ceil(x.shape[3] * scale_factor[1])
    # 计算输出宽度
    out_w = math.ceil(x.shape[4] * scale_factor[2])

    # 计算每个分块的最大通道数，以避免超出元素限制
    # 每个通道的元素数
    elements_per_channel = out_d * out_h * out_w
    # 计算最大通道数
    max_channels = max_elements_per_chunk // (x.shape[0] * elements_per_channel)

    # 使用最大通道数与输入通道数中较小的一个作为分块大小
    chunk_size = min(max_channels, x.shape[1])

    # 确保每个分块至少有一个通道
    chunk_size = max(1, chunk_size)
    if VERBOSE:
        print(f"Input channels: {x.shape[1]}")
        print(f"Chunk size: {chunk_size}")
        print(f"max_channels: {max_channels}")
        print(f"num_chunks: {math.ceil(x.shape[1] / chunk_size)}")

    # 初始化分块列表
    chunks = []
    for i in range(0, x.shape[1], chunk_size):
        # 分块起始索引
        start_idx = i

        # 分块结束索引
        end_idx = min(i + chunk_size, x.shape[1])

        # 获取当前分块
        chunk = x[:, start_idx:end_idx, :, :, :]

        # 对当前分块进行插值
        interpolated_chunk = F.interpolate(chunk, scale_factor=scale_factor, mode="nearest")

        # 将插值后的分块添加到列表中
        chunks.append(interpolated_chunk)

    if not chunks:
        raise ValueError(f"No chunks were generated. Input shape: {x.shape}")

    # 沿通道维度拼接所有分块
    return torch.cat(chunks, dim=1)


def test_chunked_interpolate():
    """
    测试函数，用于验证 chunked_interpolate 函数的正确性。
    """
    # 测试用例 1: 基本上采样，使用 scale_factor
    x1 = torch.randn(2, 16, 16, 32, 32).cuda()  # 创建一个随机输入张量
    scale_factor = (2.0, 2.0, 2.0)  # 设置缩放因子
    assert torch.allclose(
        chunked_interpolate(x1, scale_factor=scale_factor), F.interpolate(x1, scale_factor=scale_factor, mode="nearest")
    )

    # 测试用例 3: 使用 scale_factor 进行下采样
    x3 = torch.randn(2, 16, 32, 64, 64).cuda()
    scale_factor = (0.5, 0.5, 0.5)
    assert torch.allclose(
        chunked_interpolate(x3, scale_factor=scale_factor), F.interpolate(x3, scale_factor=scale_factor, mode="nearest")
    )

    # 测试用例 4: 每个维度使用不同的缩放因子
    x4 = torch.randn(2, 16, 16, 32, 32).cuda()
    scale_factor = (2.0, 1.5, 1.5)
    assert torch.allclose(
        chunked_interpolate(x4, scale_factor=scale_factor), F.interpolate(x4, scale_factor=scale_factor, mode="nearest")
    )

    # 测试用例 5: 大尺寸输入张量
    x5 = torch.randn(2, 16, 64, 128, 128).cuda()
    scale_factor = (2.0, 2.0, 2.0)
    assert torch.allclose(
        chunked_interpolate(x5, scale_factor=scale_factor), F.interpolate(x5, scale_factor=scale_factor, mode="nearest")
    )

    # 测试用例 7: 分块大小等于输入深度
    x7 = torch.randn(2, 16, 8, 32, 32).cuda()
    scale_factor = (2.0, 2.0, 2.0)
    assert torch.allclose(
        chunked_interpolate(x7, scale_factor=scale_factor), F.interpolate(x7, scale_factor=scale_factor, mode="nearest")
    )

    # 测试用例 8: 单通道输入
    x8 = torch.randn(2, 1, 16, 32, 32).cuda()
    scale_factor = (2.0, 2.0, 2.0)
    assert torch.allclose(
        chunked_interpolate(x8, scale_factor=scale_factor), F.interpolate(x8, scale_factor=scale_factor, mode="nearest")
    )

    # 测试用例 9: 最小批量大小
    x9 = torch.randn(1, 16, 32, 64, 64).cuda()
    scale_factor = (0.5, 0.5, 0.5)
    assert torch.allclose(
        chunked_interpolate(x9, scale_factor=scale_factor), F.interpolate(x9, scale_factor=scale_factor, mode="nearest")
    )

    # 测试用例 10: 非 2 的幂次方维度
    x10 = torch.randn(2, 16, 15, 31, 31).cuda()
    scale_factor = (2.0, 2.0, 2.0)
    assert torch.allclose(
        chunked_interpolate(x10, scale_factor=scale_factor),
        F.interpolate(x10, scale_factor=scale_factor, mode="nearest"),
    )


def get_same_padding(kernel_size: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
    """
    计算用于保持输入和输出尺寸相同的填充大小。

    参数:
        kernel_size (Union[int, Tuple[int, ...]]): 卷积核大小，可以是单个整数或整数元组。

    返回:
        Union[int, Tuple[int, ...]]: 填充大小，可以是单个整数或整数元组。
    """
    if isinstance(kernel_size, tuple):
        # 递归计算每个维度的填充大小
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        # 返回填充大小
        return kernel_size // 2


def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[list[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    """
    对输入张量进行插值调整尺寸。

    参数:
        x (torch.Tensor): 输入张量。
        size (Optional[Any], optional): 输出尺寸。默认为 None。
        scale_factor (Optional[List[float]], optional): 缩放因子。默认为 None。
        mode (str, optional): 插值模式。默认为 "bicubic"。可选模式包括 "bilinear", "bicubic", "nearest", "area"。
        align_corners (Optional[bool], optional): 是否对齐角点。默认为 False。

    返回:
        torch.Tensor: 调整尺寸后的张量。

    异常:
        NotImplementedError: 如果插值模式不被支持，则抛出异常。
    """
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def build_kwargs_from_config(config: dict, target_func: Callable) -> dict[str, Any]:
    """
    从配置字典中构建适用于目标函数的关键字参数。

    参数:
        config (dict): 配置字典。
        target_func (Callable): 目标函数。

    返回:
        dict[str, Any]: 适用于目标函数的关键字参数。
    """
    # 获取目标函数的参数列表
    valid_keys = list(signature(target_func).parameters)
    # 初始化关键字参数字典
    kwargs = {}
    for key in config:
        if key in valid_keys:
            # 如果配置中的键在目标函数参数中，则添加到关键字参数字典中
            kwargs[key] = config[key]
    # 返回关键字参数字典
    return kwargs


if __name__ == "__main__":
    test_chunked_interpolate()
