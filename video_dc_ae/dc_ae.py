from dataclasses import dataclass, field
from typing import Any, Optional
import torch
import torch.nn as nn
from omegaconf import MISSING, OmegaConf
from torch import Tensor

from acceleration.checkpoint import auto_grad_checkpoint

from utils import init_modules
from act import build_act
from norm import build_norm
from ops import (
    ChannelDuplicatingPixelShuffleUpSampleLayer,
    ConvLayer,
    ConvPixelShuffleUpSampleLayer,
    ConvPixelUnshuffleDownSampleLayer,
    EfficientViTBlock,
    IdentityLayer,
    InterpolateConvUpSampleLayer,
    OpSequential,
    PixelUnshuffleChannelAveragingDownSampleLayer,
    ResBlock,
    ResidualBlock,
)


# 定义一个列表，包含所有公开的接口名称
__all__ = ["DCAE", "dc_ae_f32"]


@dataclass
class EncoderConfig:
    """
    EncoderConfig 类用于配置编码器的参数。

    参数:
        in_channels (int): 输入数据的通道数。默认为 MISSING，需要在实例化时提供。
        latent_channels (int): 潜在空间的通道数。默认为 MISSING，需要在实例化时提供。
        width_list (tuple[int, ...]): 每个分辨率级别中卷积层的通道数列表。默认为 (128, 256, 512, 512, 1024, 1024)。
        depth_list (tuple[int, ...]): 每个分辨率级别中残差块的深度列表。默认为 (2, 2, 2, 2, 2, 2)。
        block_type (Any): 残差块的类型。默认为 "ResBlock"。
        norm (str): 归一化层的类型。默认为 "rms2d"。
        act (str): 激活函数的类型。默认为 "silu"。
        downsample_block_type (str): 下采样块的类型。默认为 "ConvPixelUnshuffle"。
        downsample_match_channel (bool): 下采样时是否匹配通道数。默认为 True。
        downsample_shortcut (Optional[str]): 下采样捷径连接的策略。默认为 "averaging"。
        out_norm (Optional[str]): 输出层的归一化类型。默认为 None。
        out_act (Optional[str]): 输出层的激活函数类型。默认为 None。
        out_shortcut (Optional[str]): 输出层捷径连接的策略。默认为 "averaging"。
        double_latent (bool): 是否使用双倍潜在空间。默认为 False。
        is_video (bool): 是否处理视频数据。默认为 False。
        temporal_downsample (tuple[bool, ...]): 时间下采样的配置。默认为 ()。
    """
    in_channels: int = MISSING
    latent_channels: int = MISSING
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: str = "rms2d"
    act: str = "silu"
    downsample_block_type: str = "ConvPixelUnshuffle"
    downsample_match_channel: bool = True
    downsample_shortcut: Optional[str] = "averaging"
    out_norm: Optional[str] = None
    out_act: Optional[str] = None
    out_shortcut: Optional[str] = "averaging"
    double_latent: bool = False
    is_video: bool = False
    temporal_downsample: tuple[bool, ...] = ()


@dataclass
class DecoderConfig:
    """
    DecoderConfig 类用于配置解码器的参数。

    参数:
        in_channels (int): 输入数据的通道数。默认为 MISSING，需要在实例化时提供。
        latent_channels (int): 潜在空间的通道数。默认为 MISSING，需要在实例化时提供。
        in_shortcut (Optional[str]): 输入层捷径连接的策略。默认为 "duplicating"。
        width_list (tuple[int, ...]): 每个分辨率级别中卷积层的通道数列表。默认为 (128, 256, 512, 512, 1024, 1024)。
        depth_list (tuple[int, ...]): 每个分辨率级别中残差块的深度列表。默认为 (2, 2, 2, 2, 2, 2)。
        block_type (Any): 残差块的类型。默认为 "ResBlock"。
        norm (Any): 归一化层的类型。默认为 "rms2d"。
        act (Any): 激活函数的类型。默认为 "silu"。
        upsample_block_type (str): 上采样块的类型。默认为 "ConvPixelShuffle"。
        upsample_match_channel (bool): 上采样时是否匹配通道数。默认为 True。
        upsample_shortcut (str): 上采样捷径连接的策略。默认为 "duplicating"。
        out_norm (str): 输出层的归一化类型。默认为 "rms2d"。
        out_act (str): 输出层的激活函数类型。默认为 "relu"。
        is_video (bool): 是否处理视频数据。默认为 False。
        temporal_upsample (tuple[bool, ...]): 时间上采样的配置。默认为 ()。
    """
    in_channels: int = MISSING
    latent_channels: int = MISSING
    in_shortcut: Optional[str] = "duplicating"
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: Any = "rms2d"
    act: Any = "silu"
    upsample_block_type: str = "ConvPixelShuffle"
    upsample_match_channel: bool = True
    upsample_shortcut: str = "duplicating"
    out_norm: str = "rms2d"
    out_act: str = "relu"
    is_video: bool = False
    temporal_upsample: tuple[bool, ...] = ()


@dataclass
class DCAEConfig:
    """
    DCAEConfig 类用于配置深度卷积自编码器（DCAE）的参数。

    参数:
        in_channels (int, optional): 输入数据的通道数。默认为 3。
        latent_channels (int, optional): 潜在空间的通道数。默认为 32。
        time_compression_ratio (int, optional): 时间压缩比率。默认为 1。
        spatial_compression_ratio (int, optional): 空间压缩比率。默认为 32。
        encoder (EncoderConfig, optional): 编码器的配置。默认为 EncoderConfig 实例，in_channels 和 latent_channels 从 DCAEConfig 中获取。
        decoder (DecoderConfig, optional): 解码器的配置。默认为 DecoderConfig 实例，in_channels 和 latent_channels 从 DCAEConfig 中获取。
        use_quant_conv (bool, optional): 是否使用量化卷积。默认为 False。
        pretrained_path (Optional[str], optional): 预训练模型的路径。默认为 None。
        pretrained_source (str, optional): 预训练模型的来源。默认为 "dc-ae"。
        scaling_factor (Optional[float], optional): 缩放因子。默认为 None。
        is_image_model (bool, optional): 是否为图像模型。默认为 False。
        is_training (bool, optional): 是否处于训练模式。默认为 False。
        use_spatial_tiling (bool, optional): 是否使用空间切片。默认为 False。
        use_temporal_tiling (bool, optional): 是否使用时间切片。默认为 False。
        spatial_tile_size (int, optional): 空间切片的大小。默认为 256。
        temporal_tile_size (int, optional): 时间切片的大小。默认为 32。
        tile_overlap_factor (float, optional): 切片重叠因子。默认为 0.25。
    """
    in_channels: int = 3
    latent_channels: int = 32
    time_compression_ratio: int = 1
    spatial_compression_ratio: int = 32
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(in_channels="${..in_channels}", latent_channels="${..latent_channels}")
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderConfig(in_channels="${..in_channels}", latent_channels="${..latent_channels}")
    )
    use_quant_conv: bool = False

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"

    scaling_factor: Optional[float] = None
    is_image_model: bool = False

    is_training: bool = False  # NOTE: set to True in vae train config

    use_spatial_tiling: bool = False
    use_temporal_tiling: bool = False
    spatial_tile_size: int = 256
    temporal_tile_size: int = 32
    tile_overlap_factor: float = 0.25
    

def build_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str], is_video: bool
) -> nn.Module:
    """
    根据块类型和其他参数构建神经网络块。

    参数:
        block_type (str): 块的类型，如 "ResBlock", "EViT_GLU", "EViTS5_GLU" 等。
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        norm (Optional[str]): 归一化层的类型，如果不需要归一化则为 None。
        act (Optional[str]): 激活函数的类型，如果不需要激活函数则为 None。
        is_video (bool): 是否处理视频数据。

    返回:
        nn.Module: 构建好的神经网络块。

    异常:
        ValueError: 如果 block_type 不被支持，则抛出异常。
        AssertionError: 如果 in_channels 不等于 out_channels（对于某些块类型），则抛出异常。
    """
    if block_type == "ResBlock":
        assert in_channels == out_channels
        # 创建主残差块，参数包括通道数、卷积核大小、步幅、是否使用偏置、归一化和激活函数等
        main_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            is_video=is_video,
        )
        # 将主残差块与恒等映射（IdentityLayer）结合，形成完整的残差块
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_type == "EViT_GLU":
        assert in_channels == out_channels
        # 创建 EfficientViT 块，参数包括通道数、归一化层、激活函数、局部模块类型、尺度等
        block = EfficientViTBlock(
            in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(), is_video=is_video
        )
    elif block_type == "EViTS5_GLU":
        assert in_channels == out_channels
        # 创建 EfficientViT 块，参数包括通道数、归一化层、激活函数、局部模块类型、尺度等，尺度为 (5,)
        block = EfficientViTBlock(
            in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,), is_video=is_video
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported")
    # 返回构建好的块
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int, is_video: bool
) -> list[nn.Module]:
    """
    构建一个阶段的主干模块列表。

    参数:
        width (int): 块的宽度（通道数）。
        depth (int): 块的深度（重复次数）。
        block_type (Union[str, List[str]]): 块类型，可以是单个字符串或深度长度的字符串列表。
        norm (str): 归一化层的类型。
        act (str): 激活函数的类型。
        input_width (int): 输入的宽度（通道数）。
        is_video (bool): 是否处理视频数据。

    返回:
        List[nn.Module]: 构建好的模块列表。

    异常:
        AssertionError: 如果 block_type 不是字符串或列表，或者深度与列表长度不匹配，则抛出异常。
    """
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    # 初始化阶段模块列表
    stage = []
    for d in range(depth):
        # 确定当前块的类型，如果是列表，则根据索引选择；否则使用统一的 block_type
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        # 构建当前块，参数包括块类型、输入通道数、输出通道数、归一化层、激活函数等
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width, # 如果不是第一个块，则输入通道数为 width；否则为 input_width
            out_channels=width,
            norm=norm,
            act=act,
            is_video=is_video,
        )
        stage.append(block)
    return stage


def build_downsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    is_video: bool,
    temporal_downsample: bool = False,
) -> nn.Module:
    """
    构建下采样块。空间下采样始终执行，时间下采样是可选的。

    参数:
        block_type (str): 块类型，如 "Conv", "ConvPixelUnshuffle" 等。
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        shortcut (Optional[str]): 捷径连接的策略，如果不需要捷径连接则为 None。
        is_video (bool): 是否处理视频数据。
        temporal_downsample (bool, optional): 是否进行时间下采样。默认为 False。

    返回:
        nn.Module: 构建好的下采样块。

    异常:
        NotImplementedError: 如果 block_type 为 "ConvPixelUnshuffle" 且处理视频数据，则抛出异常。
        ValueError: 如果 block_type 或 shortcut 不被支持，则抛出异常。
    """

    if block_type == "Conv":
        if is_video:
            if temporal_downsample:
                # 对于视频数据且进行时间下采样，步幅为 (2, 2, 2)
                stride = (2, 2, 2)
            else:
                # 对于视频数据且不进行时间下采样，步幅为 (1, 2, 2)
                stride = (1, 2, 2)
        else:
            # 对于非视频数据，步幅为 2
            stride = 2
        # 创建卷积层，参数包括输入通道数、输出通道数、卷积核大小、步幅等
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            use_bias=True,
            norm=None,
            act_func=None,
            is_video=is_video,
        )
    elif block_type == "ConvPixelUnshuffle":
        if is_video:
            raise NotImplementedError("ConvPixelUnshuffle downsample is not supported for video")
        # 创建 ConvPixelUnshuffle 下采样层，参数包括输入通道数、输出通道数、卷积核大小和下采样因子
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        # 如果捷径连接的策略是 "averaging"，则创建 PixelUnshuffleChannelAveragingDownSampleLayer
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2, temporal_downsample=temporal_downsample
        )
        # 将主块与捷径块结合，形成残差块
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    # 返回构建好的下采样块
    return block


def build_upsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    is_video: bool,
    temporal_upsample: bool = False,
) -> nn.Module:
    """
    构建上采样块。

    参数:
        block_type (str): 上采样块的类型，如 "ConvPixelShuffle", "InterpolateConv" 等。
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        shortcut (Optional[str]): 捷径连接的策略，如果不需要捷径连接则为 None。
        is_video (bool): 是否处理视频数据。
        temporal_upsample (bool, optional): 是否进行时间上采样。默认为 False。

    返回:
        nn.Module: 构建好的上采样块。

    异常:
        NotImplementedError: 如果 block_type 为 "ConvPixelShuffle" 且处理视频数据，则抛出异常。
        ValueError: 如果 block_type 或 shortcut 不被支持，则抛出异常。
    """
    if block_type == "ConvPixelShuffle":
        if is_video:
            # 对于视频数据，ConvPixelShuffle 上采样未实现
            raise NotImplementedError("ConvPixelShuffle upsample is not supported for video")
        # 创建 ConvPixelShuffle 上采样层，参数包括输入通道数、输出通道数、卷积核大小和上采样因子
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    elif block_type == "InterpolateConv":
        # 创建 InterpolateConv 上采样层，参数包括输入通道数、输出通道数、卷积核大小、上采样因子、视频标志和时间上采样标志
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            is_video=is_video,
            temporal_upsample=temporal_upsample,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2, temporal_upsample=temporal_upsample
        )
        # 将主块与捷径块结合，形成残差块
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    # 返回构建好的上采样块
    return block


def build_encoder_project_in_block(
    in_channels: int, out_channels: int, factor: int, downsample_block_type: str, is_video: bool
):
    """
    构建编码器 project_in 块。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        factor (int): 下采样因子。
        downsample_block_type (str): 下采样块的类型。
        is_video (bool): 是否处理视频数据。

    返回:
        nn.Module: 构建好的编码器 project_in 块。

    异常:
        NotImplementedError: 如果处理视频数据且因子为 2，则抛出异常。
        ValueError: 如果下采样因子不被支持，则抛出异常。
    """
    if factor == 1:
        # 如果因子为 1，则创建普通的卷积层，不进行下采样
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            is_video=is_video,
        )
    elif factor == 2:
        if is_video:
            # 对于视频数据，下采样未实现
            raise NotImplementedError("Downsample during project_in is not supported for video")
        # 如果因子为 2，则创建下采样块
        block = build_downsample_block(
            block_type=downsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
        )
    else:
        raise ValueError(f"downsample factor {factor} is not supported for encoder project in")
    # 返回构建好的块
    return block


def build_encoder_project_out_block(
    in_channels: int,
    out_channels: int,
    norm: Optional[str],
    act: Optional[str],
    shortcut: Optional[str],
    is_video: bool,
):
    """
    构建编码器 project_out 块。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        norm (Optional[str]): 归一化层的类型，如果不需要归一化则为 None。
        act (Optional[str]): 激活函数的类型，如果不需要激活函数则为 None。
        shortcut (Optional[str]): 捷径连接的策略，如果不需要捷径连接则为 None。
        is_video (bool): 是否处理视频数据。

    返回:
        nn.Module: 构建好的编码器 project_out 块。
    """
    # 使用 OpSequential 组合归一化、激活函数和卷积层
    block = OpSequential(
        [
            build_norm(norm),
            build_act(act),
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
                is_video=is_video,
            ),
        ]
    )
    if shortcut is None:
        # 如果不需要捷径连接，则不执行任何操作
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        # 将主块与捷径块结合，形成残差块
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for encoder project out")
    # 返回构建好的块
    return block


def build_decoder_project_in_block(in_channels: int, out_channels: int, shortcut: Optional[str], is_video: bool):
    """
    构建解码器 project_in 块。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        shortcut (Optional[str]): 捷径连接的策略，如果不需要捷径连接则为 None。
        is_video (bool): 是否处理视频数据。

    返回:
        nn.Module: 构建好的解码器 project_in 块。
    """
    # 创建卷积层
    block = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        use_bias=True,
        norm=None,
        act_func=None,
        is_video=is_video,
    )
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        # 将主块与捷径块结合，形成残差块
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for decoder project in")
    # 返回构建好的块
    return block


def build_decoder_project_out_block(
    in_channels: int,
    out_channels: int,
    factor: int,
    upsample_block_type: str,
    norm: Optional[str],
    act: Optional[str],
    is_video: bool,
):
    """
    构建解码器 project_out 块。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        factor (int): 上采样因子。
        upsample_block_type (str): 上采样块的类型。
        norm (Optional[str]): 归一化层的类型，如果不需要归一化则为 None。
        act (Optional[str]): 激活函数的类型，如果不需要激活函数则为 None。
        is_video (bool): 是否处理视频数据。

    返回:
        nn.Module: 构建好的解码器 project_out 块。
    """
    layers: list[nn.Module] = [
        build_norm(norm, in_channels), # 构建归一化层
        build_act(act), # 构建激活函数
    ]
    if factor == 1:
        # 如果上采样因子为 1，则添加一个卷积层，不进行上采样
        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
                is_video=is_video,
            )
        )
    elif factor == 2:
        if is_video:
            # 对于视频数据，上采样未实现
            raise NotImplementedError("Upsample during project_out is not supported for video")
        # 如果上采样因子为 2，则添加一个上采样块
        layers.append(
            build_upsample_block(
                block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
            )
        )
    else:
        # 如果上采样因子不被支持，则抛出异常
        raise ValueError(f"upsample factor {factor} is not supported for decoder project out")
    # 将所有层组合成一个顺序模块并返回
    return OpSequential(layers)


class Encoder(nn.Module):
    """
    Encoder 类实现了编码器部分。

    参数:
        cfg (EncoderConfig): 编码器的配置参数。
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        # 计算阶段数
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )

        # 构建 project_in 块，用于将输入数据投影到编码器的初始通道数
        self.project_in = build_encoder_project_in_block(
            in_channels=cfg.in_channels,
            out_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            factor=1 if cfg.depth_list[0] > 0 else 2,
            downsample_block_type=cfg.downsample_block_type,
            is_video=cfg.is_video,
        )

        # 初始化阶段列表
        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            # 确定当前阶段的块类型
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            # 构建当前阶段的主干模块列表
            stage = build_stage_main(
                width=width,
                depth=depth,
                block_type=block_type,
                norm=cfg.norm,
                act=cfg.act,
                input_width=width,
                is_video=cfg.is_video,
            )

            if stage_id < num_stages - 1 and depth > 0:
                # 如果不是最后一个阶段且深度大于 0，则添加下采样块
                downsample_block = build_downsample_block(
                    block_type=cfg.downsample_block_type,
                    in_channels=width,
                    out_channels=cfg.width_list[stage_id + 1] if cfg.downsample_match_channel else width,
                    shortcut=cfg.downsample_shortcut,
                    is_video=cfg.is_video,
                    temporal_downsample=cfg.temporal_downsample[stage_id] if cfg.temporal_downsample != [] else False,
                )
                stage.append(downsample_block)
            # 将当前阶段的主干模块列表封装成 OpSequential 并添加到阶段列表中
            self.stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        # 构建 project_out 块，用于将编码器的输出投影到潜在空间
        self.project_out = build_encoder_project_out_block(
            in_channels=cfg.width_list[-1],
            out_channels=2 * cfg.latent_channels if cfg.double_latent else cfg.latent_channels,
            norm=cfg.out_norm,
            act=cfg.out_act,
            shortcut=cfg.out_shortcut,
            is_video=cfg.is_video,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，实现编码过程。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 编码后的潜在空间张量。
        """
        # 应用 project_in 块
        x = self.project_in(x)
        # x = auto_grad_checkpoint(self.project_in, x)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            # 应用阶段模块，并使用 auto_grad_checkpoint 进行梯度检查点
            x = auto_grad_checkpoint(stage, x)
        # x = self.project_out(x)
        # 应用 project_out 块，并使用 auto_grad_checkpoint 进行梯度检查点
        x = auto_grad_checkpoint(self.project_out, x)
        return x


class Decoder(nn.Module):
    """
    Decoder 类实现了解码器部分。

    参数:
        cfg (DecoderConfig): 解码器的配置参数。
    """
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        # 计算阶段数
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.act, str) or (isinstance(cfg.act, list) and len(cfg.act) == num_stages)

        # 构建 project_in 块，用于将潜在空间的数据投影到解码器的初始通道数
        self.project_in = build_decoder_project_in_block(
            in_channels=cfg.latent_channels,
            out_channels=cfg.width_list[-1],
            shortcut=cfg.in_shortcut,
            is_video=cfg.is_video,
        )

        # 初始化阶段列表
        self.stages: list[OpSequential] = []
        # 反向遍历宽度和深度列表，从最高分辨率到最低分辨率构建阶段
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            # 初始化当前阶段的模块列表
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                # 如果不是最后一个阶段且深度大于 0，则构建上采样块
                upsample_block = build_upsample_block(
                    block_type=cfg.upsample_block_type,
                    in_channels=cfg.width_list[stage_id + 1],
                    out_channels=width if cfg.upsample_match_channel else cfg.width_list[stage_id + 1],
                    shortcut=cfg.upsample_shortcut,
                    is_video=cfg.is_video,
                    temporal_upsample=cfg.temporal_upsample[stage_id] if cfg.temporal_upsample != [] else False,
                )
                # 将上采样块添加到当前阶段
                stage.append(upsample_block)

            # 确定当前阶段的块类型、归一化和激活函数
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            act = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            # 构建当前阶段的主干模块列表，并将其添加到当前阶段
            stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=block_type,
                    norm=norm,
                    act=act,
                    input_width=(
                        width if cfg.upsample_match_channel else cfg.width_list[min(stage_id + 1, num_stages - 1)]
                    ),
                    is_video=cfg.is_video,
                )
            )
            # 将当前阶段封装成 OpSequential 并插入到阶段列表的前面，以保持顺序一致
            self.stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        # 构建 project_out 块，用于将解码器的输出投影到输出空间
        self.project_out = build_decoder_project_out_block(
            in_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            out_channels=cfg.in_channels,
            factor=1 if cfg.depth_list[0] > 0 else 2,
            upsample_block_type=cfg.upsample_block_type,
            norm=cfg.out_norm,
            act=cfg.out_act,
            is_video=cfg.is_video,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，实现解码过程。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 解码后的输出张量。
        """
        # 应用 project_in 块，并使用 auto_grad_checkpoint 进行梯度检查点
        x = auto_grad_checkpoint(self.project_in, x)
        for stage in reversed(self.stages):
            if len(stage.op_list) == 0:
                continue
            # x = stage(x)
            # 应用当前阶段，并使用 auto_grad_checkpoint 进行梯度检查点
            x = auto_grad_checkpoint(stage, x)

        if self.disc_off_grad_ckpt:
            # 如果 disc_off_grad_ckpt 为 True，则直接应用 project_out 块
            x = self.project_out(x)
        else:
            # 否则，使用 auto_grad_checkpoint 进行梯度检查点
            x = auto_grad_checkpoint(self.project_out, x)
        return x


class DCAE(nn.Module):
    """
    DCAE 类实现了深度卷积自编码器（DCAE）。

    参数:
        cfg (DCAEConfig): DCAE 的配置参数。
    """
    def __init__(self, cfg: DCAEConfig):
        super().__init__()
        self.cfg = cfg
        # 初始化编码器
        self.encoder = Encoder(cfg.encoder)
        # 初始化解码器
        self.decoder = Decoder(cfg.decoder)
        # 设置缩放因子
        self.scaling_factor = cfg.scaling_factor
        # 设置时间压缩比率
        self.time_compression_ratio = cfg.time_compression_ratio
        # 设置空间压缩比率
        self.spatial_compression_ratio = cfg.spatial_compression_ratio
        # 设置是否使用空间切片
        self.use_spatial_tiling = cfg.use_spatial_tiling
        # 设置是否使用时间切片
        self.use_temporal_tiling = cfg.use_temporal_tiling
        # 设置空间切片大小
        self.spatial_tile_size = cfg.spatial_tile_size
        # 设置时间切片大小
        self.temporal_tile_size = cfg.temporal_tile_size
        assert (
            cfg.spatial_tile_size // cfg.spatial_compression_ratio
        ), f"spatial tile size {cfg.spatial_tile_size} must be divisible by spatial compression of {cfg.spatial_compression_ratio}"
        
        # 计算空间切片潜在空间大小
        self.spatial_tile_latent_size = cfg.spatial_tile_size // cfg.spatial_compression_ratio
        assert (
            cfg.temporal_tile_size // cfg.time_compression_ratio
        ), f"temporal tile size {cfg.temporal_tile_size} must be divisible by temporal compression of {cfg.time_compression_ratio}"
        
        # 计算时间切片潜在空间大小
        self.temporal_tile_latent_size = cfg.temporal_tile_size // cfg.time_compression_ratio
        # 设置切片重叠因子
        self.tile_overlap_factor = cfg.tile_overlap_factor
        if self.cfg.pretrained_path is not None:
            # 如果提供了预训练路径，则加载预训练模型
            self.load_model()

        self.to(torch.float32)
        # 初始化模型参数，使用截断正态分布
        init_modules(self, init_type="trunc_normal")

    def load_model(self):
        """
        加载预训练模型。
        """
        if self.cfg.pretrained_source == "dc-ae":
            # 从指定路径加载预训练模型的状态字典
            state_dict = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)["state_dict"]
            # 加载状态字典到模型中
            self.load_state_dict(state_dict)
        else:
            raise NotImplementedError

    def get_last_layer(self):
        """
        获取解码器的最后一层权重。

        返回:
            torch.Tensor: 解码器最后一层的权重。
        """
        # 返回解码器 project_out 块的第三个操作中的卷积权重
        return self.decoder.project_out.op_list[2].conv.weight

    # @property
    # def spatial_compression_ratio(self) -> int:
    #     return 2 ** (self.decoder.num_stages - 1)

    def encode_single(self, x: torch.Tensor, is_video_encoder: bool = False) -> torch.Tensor:
        """
        对单个输入样本进行编码。

        参数:
            x (torch.Tensor): 输入张量，形状为 (1, C, ...) 或 (1, T, C, H, W)。
            is_video_encoder (bool, optional): 是否为视频编码器。默认为 False。

        返回:
            torch.Tensor: 编码后的潜在空间张量。
        """
        assert x.shape[0] == 1
        # 判断输入是否为视频数据（5 维）
        is_video = x.dim() == 5
        if is_video and not is_video_encoder:
            # 解包视频数据的维度
            b, c, f, h, w = x.shape
            # 重排维度，将时间维度与批量维度合并
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        # 通过编码器进行编码
        z = self.encoder(x)

        if is_video and not is_video_encoder:
            # 如果是视频数据且不是视频编码器，则调整潜在空间的维度
            z = z.unsqueeze(dim=0).permute(0, 2, 1, 3, 4)

        if self.scaling_factor is not None:
            # 如果设置了缩放因子，则对潜在空间进行缩放
            z = z / self.scaling_factor
        # 返回编码后的潜在空间张量
        return z

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量进行编码，根据训练状态决定是否使用编码器。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 编码后的潜在空间张量。
        """
        if self.cfg.is_training:
            # 如果处于训练状态，则直接使用编码器
            return self.encoder(x)
        # 判断是否为视频编码器
        is_video_encoder = self.encoder.cfg.is_video if self.encoder.cfg.is_video is not None else False
        # 初始化返回列表
        x_ret = []
        for i in range(x.shape[0]):
            # 对每个样本进行编码
            x_ret.append(self.encode_single(x[i : i + 1], is_video_encoder))
        # 将所有编码结果拼接起来
        return torch.cat(x_ret, dim=0)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """
        在垂直方向上混合两个张量。

        参数:
            a (torch.Tensor): 第一个输入张量。
            b (torch.Tensor): 第二个输入张量。
            blend_extent (int): 混合范围。

        返回:
            torch.Tensor: 混合后的张量。
        """
        # 确定混合范围
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            # 在垂直方向上逐步混合
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        # 返回混合后的张量
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """
        在水平方向上混合两个张量。

        参数:
            a (torch.Tensor): 第一个输入张量。
            b (torch.Tensor): 第二个输入张量。
            blend_extent (int): 混合范围。

        返回:
            torch.Tensor: 混合后的张量。
        """
        # 确定混合范围
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            # 在水平方向上逐步混合
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        # 返回混合后的张量
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """
        在时间方向上混合两个张量。

        参数:
            a (torch.Tensor): 第一个输入张量。
            b (torch.Tensor): 第二个输入张量。
            blend_extent (int): 混合范围。

        返回:
            torch.Tensor: 混合后的张量。
        """
        # 确定混合范围
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            # 在时间方向上逐步混合
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        # 返回混合后的张量
        return b

    def spatial_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量进行空间切片编码。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 编码后的张量。
        """
        # 计算切片之间的净尺寸
        net_size = int(self.spatial_tile_size * (1 - self.tile_overlap_factor))
        # 计算混合范围
        blend_extent = int(self.spatial_tile_latent_size * self.tile_overlap_factor)
        # 计算每行的限制
        row_limit = self.spatial_tile_latent_size - blend_extent

        # 将视频分割成切片并分别编码
        rows = [] # 初始化行列表
        for i in range(0, x.shape[-2], net_size):
            # 初始化当前行的切片列表
            row = []
            for j in range(0, x.shape[-1], net_size):
                # 分割切片
                tile = x[:, :, :, i : i + self.spatial_tile_size, j : j + self.spatial_tile_size]
                # 对切片进行编码
                tile = self._encode(tile)
                # 将编码后的切片添加到当前行
                row.append(tile)
            # 将当前行添加到行列表
            rows.append(row)
        # 初始化结果行列表
        result_rows = []
        for i, row in enumerate(rows):
            # 初始化当前结果行
            result_row = []
            for j, tile in enumerate(row):
                # 将上面的切片和左边的切片混合到当前切片
                if i > 0:
                    # 垂直方向混合
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    # 水平方向混合
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                # 添加处理后的切片到结果行
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            # 将结果行拼接起来
            result_rows.append(torch.cat(result_row, dim=-1))
        # 将所有结果行拼接起来
        return torch.cat(result_rows, dim=-2)

    def temporal_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量进行时间切片编码。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 编码后的张量。
        """
        # 计算切片之间的重叠尺寸
        overlap_size = int(self.temporal_tile_size * (1 - self.tile_overlap_factor))
        # 计算混合范围
        blend_extent = int(self.temporal_tile_latent_size * self.tile_overlap_factor)
        # 计算时间限制
        t_limit = self.temporal_tile_latent_size - blend_extent

        # 将视频分割成切片并分别编码
        row = []
        for i in range(0, x.shape[2], overlap_size):
            # 分割切片
            tile = x[:, :, i : i + self.temporal_tile_size, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.spatial_tile_size or tile.shape[-2] > self.spatial_tile_size
            ):
                # 如果使用空间切片且切片尺寸超过限制，则进行空间切片编码
                tile = self.spatial_tiled_encode(tile)
            else:
                # 否则，直接编码
                tile = self._encode(tile)
            # 将编码后的切片添加到行列表
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                # 时间方向混合
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            # 添加处理后的切片到结果行
            result_row.append(tile[:, :, :t_limit, :, :])
        # 将结果行拼接起来
        return torch.cat(result_row, dim=2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量进行编码，根据切片选项决定编码方式。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 编码后的张量。
        """
        if self.use_temporal_tiling and x.shape[2] > self.temporal_tile_size:
            # 如果使用时间切片且时间维度超过限制，则进行时间切片编码
            return self.temporal_tiled_encode(x)
        elif self.use_spatial_tiling and (x.shape[-1] > self.spatial_tile_size or x.shape[-2] > self.spatial_tile_size):
            # 如果使用空间切片且空间维度超过限制，则进行空间切片编码
            return self.spatial_tiled_encode(x)
        else:
            # 否则，直接编码
            return self._encode(x)

    def spatial_tiled_decode(self, z: torch.FloatTensor) -> torch.Tensor:
        """
        对潜在空间张量进行空间切片解码。

        参数:
            z (torch.FloatTensor): 潜在空间张量。

        返回:
            torch.Tensor: 解码后的张量。
        """
        # 计算切片之间的净尺寸
        net_size = int(self.spatial_tile_latent_size * (1 - self.tile_overlap_factor))
        # 计算混合范围
        blend_extent = int(self.spatial_tile_size * self.tile_overlap_factor)
        # 计算每行的限制
        row_limit = self.spatial_tile_size - blend_extent

        # 将 z 分割成重叠的切片并分别解码。
        # 切片之间有重叠，以避免切片之间的接缝。
        # 初始化行列表
        rows = []
        for i in range(0, z.shape[-2], net_size):
            # 初始化当前行的切片列表
            row = []
            for j in range(0, z.shape[-1], net_size):
                # 分割切片
                tile = z[:, :, :, i : i + self.spatial_tile_latent_size, j : j + self.spatial_tile_latent_size]
                # 对切片进行解码
                decoded = self._decode(tile)
                # 将解码后的切片添加到当前行
                row.append(decoded)
            # 将当前行添加到行列表
            rows.append(row)
        # 初始化结果行列表
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # 将上面的切片和左边的切片混合到当前切片
                if i > 0:
                    # 垂直方向混合
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    # 水平方向混合
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                # 添加处理后的切片到结果行
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            # 将结果行拼接起来
            result_rows.append(torch.cat(result_row, dim=-1))
        # 将所有结果行拼接起来
        return torch.cat(result_rows, dim=-2)

    def temporal_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        对潜在空间张量进行时间切片解码。

        参数:
            z (torch.Tensor): 潜在空间张量。

        返回:
            torch.Tensor: 解码后的张量。
        """
        # 计算切片之间的重叠尺寸
        overlap_size = int(self.temporal_tile_latent_size * (1 - self.tile_overlap_factor))
        # 计算混合范围
        blend_extent = int(self.temporal_tile_size * self.tile_overlap_factor)
        # 计算时间限制
        t_limit = self.temporal_tile_size - blend_extent

        # 初始化行列表
        row = []
        for i in range(0, z.shape[2], overlap_size):
            # 分割切片
            tile = z[:, :, i : i + self.temporal_tile_latent_size, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.spatial_tile_latent_size or tile.shape[-2] > self.spatial_tile_latent_size
            ):
                # 如果使用空间切片且切片尺寸超过限制，则进行空间切片解码
                decoded = self.spatial_tiled_decode(tile)
            else:
                # 否则，直接解码
                decoded = self._decode(tile)
            # 将解码后的切片添加到行列表
            row.append(decoded)
        # 初始化结果行
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                # 时间方向混合
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            # 添加处理后的切片到结果行
            result_row.append(tile[:, :, :t_limit, :, :])
        # 将结果行拼接起来
        return torch.cat(result_row, dim=2)

    def decode_single(self, z: torch.Tensor, is_video_decoder: bool = False) -> torch.Tensor:
        """
        对单个潜在空间样本进行解码。

        参数:
            z (torch.Tensor): 潜在空间张量，形状为 (1, C, ...) 或 (1, T, C, H, W)。
            is_video_decoder (bool, optional): 是否为视频解码器。默认为 False。

        返回:
            torch.Tensor: 解码后的张量。
        """
        assert z.shape[0] == 1
        # 判断输入是否为视频数据（5 维）
        is_video = z.dim() == 5
        if is_video and not is_video_decoder:
            # 解包视频数据的维度
            b, c, f, h, w = z.shape
            # 重排维度，将时间维度与批量维度合并
            z = z.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        if self.scaling_factor is not None:
            # 如果设置了缩放因子，则对潜在空间进行缩放
            z = z * self.scaling_factor

        # 通过解码器进行解码
        x = self.decoder(z)

        if is_video and not is_video_decoder:
            # 如果是视频数据且不是视频解码器，则调整解码后的张量维度
            x = x.unsqueeze(dim=0).permute(0, 2, 1, 3, 4)
        # 返回解码后的张量
        return x

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        对输入潜在空间张量进行解码，根据训练状态决定是否使用解码器。

        参数:
            z (torch.Tensor): 输入潜在空间张量。

        返回:
            torch.Tensor: 解码后的张量。
        """
        if self.cfg.is_training:
            # 如果处于训练状态，则直接使用解码器
            return self.decoder(z)
        # 判断是否为视频解码器
        is_video_decoder = self.decoder.cfg.is_video if self.decoder.cfg.is_video is not None else False
        x_ret = []
        for i in range(z.shape[0]):
            # 对每个样本进行解码
            x_ret.append(self.decode_single(z[i : i + 1], is_video_decoder))
        # 将所有解码结果拼接起来
        return torch.cat(x_ret, dim=0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        对输入潜在空间张量进行解码，根据切片选项决定解码方式。

        参数:
            z (torch.Tensor): 输入潜在空间张量。

        返回:
            torch.Tensor: 解码后的张量。
        """
        if self.use_temporal_tiling and z.shape[2] > self.temporal_tile_latent_size:
            # 如果使用时间切片且时间维度超过限制，则进行时间切片解码
            return self.temporal_tiled_decode(z)
        elif self.use_spatial_tiling and (
            z.shape[-1] > self.spatial_tile_latent_size or z.shape[-2] > self.spatial_tile_latent_size
        ):
            # 如果使用空间切片且空间维度超过限制，则进行空间切片解码
            return self.spatial_tiled_decode(z)
        else:
            return self._decode(z)

    def forward(self, x: torch.Tensor) -> tuple[Any, Tensor, dict[Any, Any]]:
        """
        前向传播方法，实现编码和解码过程。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            tuple[Any, Tensor, dict[Any, Any]]: 解码后的张量、None 和潜在空间张量。
        """
        # 获取输入张量数据类型
        x_type = x.dtype
        # 判断是否为图像模型
        is_image_model = self.cfg.__dict__.get("is_image_model", False)
        # 将输入张量移动到编码器卷积权重的数据类型
        x = x.to(self.encoder.project_in.conv.weight.dtype)

        if is_image_model:
            # 解包图像模型的维度
            b, c, _, h, w = x.shape
            # 重排维度，将时间维度与批量维度合并
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

        # 进行编码
        z = self.encode(x)
        # 进行解码
        dec = self.decode(z)

        if is_image_model:
            # 调整解码后张量的维度
            dec = dec.reshape(b, 1, c, h, w).permute(0, 2, 1, 3, 4)
            # 调整潜在空间张量的维度
            z = z.unsqueeze(dim=0).permute(0, 2, 1, 3, 4)

        # 将解码后张量转换回原始数据类型
        dec = dec.to(x_type)
        # 返回解码后张量、None 和潜在空间张量
        return dec, None, z

    def get_latent_size(self, input_size: list[int]) -> list[int]:
        """
        计算潜在空间的大小。

        参数:
            input_size (list[int]): 输入数据的大小，列表形式 [时间步长, 高度, 宽度]。

        返回:
            list[int]: 潜在空间的大小，列表形式 [时间步长, 高度, 宽度]。
        """
        # 初始化潜在空间大小列表
        latent_size = []
        # 时间步长
        latent_size.append((input_size[0] - 1) // self.time_compression_ratio + 1)
        # 高度和宽度
        for i in range(1, 3):
            latent_size.append((input_size[i] - 1) // self.spatial_compression_ratio + 1)
        # 返回潜在空间大小列表
        return latent_size


def dc_ae_f32(name: str, pretrained_path: str) -> DCAEConfig:
    """
    根据模型名称和预训练路径创建 DCAE 配置。

    参数:
        name (str): 模型名称。
        pretrained_path (str): 预训练模型的路径。

    返回:
        DCAEConfig: 配置对象，包含模型的配置参数。
    """
    if name in ["dc-ae-f32t4c128"]:
        cfg_str = (
            "time_compression_ratio=4 "
            "spatial_compression_ratio=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[2,2,2,3,3,3] "
            "encoder.downsample_block_type=Conv "
            "encoder.norm=rms3d "
            "encoder.is_video=True "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[3,3,3,3,3,3] "
            "decoder.upsample_block_type=InterpolateConv "
            "decoder.norm=rms3d decoder.act=silu decoder.out_norm=rms3d "
            "decoder.is_video=True "
            "encoder.temporal_downsample=[False,False,False,True,True,False] "
            "decoder.temporal_upsample=[False,False,False,True,True,False] "
            "latent_channels=128"
        )  # 确保最后一行没有尾随空格
    else:
        raise NotImplementedError
    
    # 使用 OmegaConf 解析配置字符串
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))

    # 将配置合并到 DCAEConfig 中
    cfg: DCAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))
    
    # 设置预训练路径
    cfg.pretrained_path = pretrained_path

    return cfg

