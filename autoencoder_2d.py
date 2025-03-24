from dataclasses import dataclass
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import silu as swish

from registry import MODELS
from utils.ckpt import load_checkpoint
from utils import DiagonalGaussianDistribution


@dataclass
class AutoEncoderConfig:
    """
    AutoEncoderConfig 类用于配置自动编码器的参数。
    
    参数:
        from_pretrained (str | None): 预训练模型的路径或标识符。如果为 None，则不使用预训练模型。
        cache_dir (str | None): 缓存目录，用于存储下载的预训练模型和缓存数据。如果为 None，则使用默认缓存目录。
        resolution (int): 输入图像的分辨率（高度和宽度的像素数）。
        in_channels (int): 输入图像的通道数（例如，RGB 图像为 3）。
        ch (int): 初始隐藏层的通道数。
        out_ch (int): 输出层的通道数。
        ch_mult (list[int]): 通道数乘数列表，用于在每个下采样阶段增加通道数。
        num_res_blocks (int): 每个分辨率级别中残差块的数目。
        z_channels (int): 潜在空间（z 空间）的通道数。
        scale_factor (float): 缩放因子，用于调整潜在空间的大小。
        shift_factor (float): 平移因子，用于调整潜在空间的位置。
        sample (bool, optional): 是否进行采样操作。默认为 True。
    """
    from_pretrained: str | None
    cache_dir: str | None
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float
    sample: bool = True


class AttnBlock(nn.Module):
    """
    AttnBlock 类实现了自注意力机制模块。
    
    参数:
        in_channels (int): 输入特征的通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()

        # 定义组归一化层
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # 定义用于计算查询（Q）、键（K）和值（V）的 1x1 卷积层
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 定义输出投影的 1x1 卷积层
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        """
        计算自注意力。
        
        参数:
            h_ (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
        
        返回:
            torch.Tensor: 注意力加权后的张量，形状为 (B, C, H, W)。
        """
        # 对输入进行归一化
        h_ = self.norm(h_)

        # 计算 Q, K, V
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 获取张量的维度
        b, c, h, w = q.shape

        # 重排张量形状以适应注意力计算
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()

        # 计算缩放点积注意力
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        # 重排回原始形状
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
        
        返回:
            torch.Tensor: 经过注意力机制处理后的张量，形状为 (B, C, H, W)。
        """
        # 残差连接：输出 = 输入 + 注意力处理后的输入
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """
    ResnetBlock 类实现了残差块模块。
    
    参数:
        in_channels (int): 输入特征的通道数。
        out_channels (int | None): 输出特征的通道数。如果为 None，则输出通道数与输入通道数相同。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        # 如果 out_channels 为 None，则设置为 in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # 定义第一个组归一化层
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        # 定义第一个 3x3 卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 定义第二个组归一化层
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        # 定义第二个 3x3 卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 如果输入和输出通道数不同，则定义一个 1x1 卷积层用于捷径连接
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播方法。
        
        参数:
            x (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 经过残差块处理后的张量。
        """
        h = x
        # 应用第一个归一化和激活函数
        h = self.norm1(h)
        # 定义激活函数
        h = swish(h)
        # 应用第一个卷积层
        h = self.conv1(h)

        # 应用第二个归一化和激活函数
        h = self.norm2(h)
        h = swish(h)
        # 应用第二个卷积层
        h = self.conv2(h)

        # 如果输入和输出通道数不同，则应用捷径连接的 1x1 卷积
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        # 残差连接：输出 = 输入 + 第二个卷积层的输出
        return x + h


class Downsample(nn.Module):
    """
    Downsample 类实现了下采样模块。
    
    参数:
        in_channels (int): 输入特征的通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 定义一个 3x3 卷积层，步幅为 2，实现下采样
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法。
        
        参数:
            x (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 经过下采样处理后的张量。
        """
        # 对输入进行填充，以适应卷积层的步幅
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        # 应用卷积层
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsample 类实现了上采样模块。
    
    参数:
        in_channels (int): 输入特征的通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 定义一个 3x3 卷积层，用于调整通道数并保持特征图大小
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法，实现上采样操作。
        
        参数:
            x (Tensor): 输入张量。
        
        返回:
            Tensor: 经过上采样处理后的张量。
        """
        # 使用最近邻插值法将特征图的空间尺寸放大 2 倍
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 应用卷积层
        return self.conv(x)


class Encoder(nn.Module):
    """
    Encoder 类实现了自动编码器中的编码器部分。
    
    参数:
        config (AutoEncoderConfig): 自动编码器的配置参数。
    """
    def __init__(self, config: AutoEncoderConfig):
        super().__init__()
        self.ch = config.ch
        # 分辨率级别数
        self.num_resolutions = len(config.ch_mult)
        # 每个分辨率级别的残差块数
        self.num_res_blocks = config.num_res_blocks
        # 输入图像的分辨率
        self.resolution = config.resolution
        # 输入图像的通道数
        self.in_channels = config.in_channels

        # 下采样部分
        # 定义输入卷积层，将输入图像转换为初始通道数
        self.conv_in = nn.Conv2d(config.in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        # 计算每个分辨率级别的通道乘数
        curr_res = config.resolution
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.in_ch_mult = in_ch_mult

        # 定义下采样模块列表
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            # 当前分辨率级别的残差块列表
            block = nn.ModuleList()
            # 当前分辨率级别的注意力模块列表
            attn = nn.ModuleList()
            # 当前分辨率级别的输入通道数
            block_in = config.ch * in_ch_mult[i_level]
            # 当前分辨率级别的输出通道数
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                # 添加残差块
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                # 更新输入通道数
                block_in = block_out
            # 定义当前分辨率级别的下采样模块
            down = nn.Module()
            # 赋值残差块列表
            down.block = block
            # 赋值注意力模块列表（当前为空）
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                # 如果不是最后一个分辨率级别，添加下采样操作
                down.downsample = Downsample(block_in)
                # 更新当前分辨率
                curr_res = curr_res // 2
            # 将当前分辨率级别的下采样模块添加到列表中
            self.down.append(down)

        # 中间部分
        self.mid = nn.Module()
        # 第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 第一个注意力模块
        self.mid.attn_1 = AttnBlock(block_in)
        # 第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # 输出部分
        # 定义组归一化层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # 定义输出卷积层，将特征图转换为潜在空间的通道数
        self.conv_out = nn.Conv2d(block_in, 2 * config.z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法，实现编码过程。
        
        参数:
            x (Tensor): 输入图像张量。
        
        返回:
            Tensor: 编码后的潜在空间张量。
        """
        # 下采样过程
        # 应用输入卷积层，并添加到列表中
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # 应用残差块
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    # 应用注意力模块
                    h = self.down[i_level].attn[i_block](h)
                # 将结果添加到列表中
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                # 应用下采样操作并添加到列表中
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 中间过程
        # 获取最后一个输出
        h = hs[-1]
        # 应用第一个残差块
        h = self.mid.block_1(h)
        # 应用第一个注意力模块
        h = self.mid.attn_1(h)
        # 应用第二个残差块
        h = self.mid.block_2(h)

        # 输出过程
        # 应用组归一化
        h = self.norm_out(h)
        # 应用激活函数
        h = swish(h)
        # 应用输出卷积层
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """
    Decoder 类实现了自动编码器中的解码器部分。
    
    参数:
        config (AutoEncoderConfig): 自动编码器的配置参数。
    """
    def __init__(self, config: AutoEncoderConfig):
        super().__init__()
        # 初始通道数
        self.ch = config.ch
        # 分辨率级别数
        self.num_resolutions = len(config.ch_mult)
        # 每个分辨率级别的残差块数
        self.num_res_blocks = config.num_res_blocks
        # 输出图像的分辨率
        self.resolution = config.resolution
        # 输入图像的通道数
        self.in_channels = config.in_channels
        # 计算上采样因子
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # 计算中间块的输入通道数
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        # 计算中间块的分辨率
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        # 定义潜在空间张量的形状
        self.z_shape = (1, config.z_channels, curr_res, curr_res)

        # 定义从潜在空间到中间块的卷积层
        self.conv_in = nn.Conv2d(config.z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间部分
        self.mid = nn.Module()
        # 第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 第一个注意力模块
        self.mid.attn_1 = AttnBlock(block_in)
        # 第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # 上采样部分
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            # 当前分辨率级别的残差块列表
            block = nn.ModuleList()
            # 当前分辨率级别的注意力模块列表
            attn = nn.ModuleList()
            # 当前分辨率级别的输出通道数
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                # 添加残差块
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                # 更新输入通道数
                block_in = block_out
            # 定义当前分辨率级别的上采样模块
            up = nn.Module()
            # 赋值残差块列表
            up.block = block
            # 赋值注意力模块列表
            up.attn = attn
            if i_level != 0:
                # 如果不是第一个分辨率级别，添加上采样操作
                up.upsample = Upsample(block_in)
                # 更新当前分辨率
                curr_res = curr_res * 2
            # 将上采样模块添加到列表的前面，以保持顺序一致
            self.up.insert(0, up) 

        # 输出部分
        # 定义组归一化层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # 定义输出卷积层，将特征图转换为输出图像的通道数
        self.conv_out = nn.Conv2d(block_in, config.out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播方法，实现解码过程。
        
        参数:
            z (Tensor): 输入的潜在空间张量。
        
        返回:
            Tensor: 解码后的输出图像张量。
        """
        # 从潜在空间到中间块
        # 应用卷积层
        h = self.conv_in(z)

        # 中间部分
        # 应用第一个残差块
        h = self.mid.block_1(h)
        # 应用第一个注意力模块
        h = self.mid.attn_1(h)
        # 应用第二个残差块
        h = self.mid.block_2(h)

        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                # 应用残差块
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    # 应用注意力模块
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                # 应用上采样操作
                h = self.up[i_level].upsample(h)

        # 输出部分
        # 应用组归一化
        h = self.norm_out(h)
        # 应用激活函数
        h = swish(h)
        # 应用输出卷积层并返回结果
        return self.conv_out(h)


class AutoEncoder(nn.Module):
    """
    AutoEncoder 类实现了自动编码器模型，包括编码器和解码器部分。
    
    参数:
        config (AutoEncoderConfig): 自动编码器的配置参数。
    """
    def __init__(self, config: AutoEncoderConfig):
        super().__init__()

        # 初始化编码器和解码器
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # 设置缩放因子和平移因子，用于潜在空间正则化
        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

        # 设置是否进行采样操作
        self.sample = config.sample

    def encode_(self, x: Tensor) -> tuple[Tensor, DiagonalGaussianDistribution]:
        """
        对输入张量进行编码，返回潜在向量和潜在分布参数。
        
        参数:
            x (Tensor): 输入张量，形状为 (B, C, T, H, W)。
        
        返回:
            Tuple[Tensor, DiagonalGaussianDistribution]: 
                - 潜在向量 z，形状为 (B, C_z, T, H_z, W_z)。
                - 潜在分布参数，DiagonalGaussianDistribution 对象。
        """
        # 获取时间维度长度
        T = x.shape[2]
        # 重排张量形状，从 (B, C, T, H, W) 变为 (B*T, C, H, W)
        x = rearrange(x, "b c t h w -> (b t) c h w")
        # 通过编码器获取参数，用于潜在分布
        params = self.encoder(x)
        # 重排回原始时间维度，从 (B*T, C_z, H_z, W_z) 变为 (B, C_z, T, H_z, W_z)
        params = rearrange(params, "(b t) c h w -> b c t h w", t=T)
        posterior = DiagonalGaussianDistribution(params)

        # 根据配置决定是采样还是使用均值作为潜在向量
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # 对潜在向量进行缩放和平移
        z = self.scale_factor * (z - self.shift_factor)
        return z, posterior

    def encode(self, x: Tensor) -> Tensor:
        """
        对输入张量进行编码，返回潜在向量。
        
        参数:
            x (Tensor): 输入张量，形状为 (B, C, T, H, W)。
        
        返回:
            Tensor: 潜在向量 z，形状为 (B, C_z, T, H_z, W_z)。
        """
        return self.encode_(x)[0]

    def decode(self, z: Tensor) -> Tensor:
        """
        对潜在向量进行解码，返回重构的输入张量。
        
        参数:
            z (Tensor): 潜在向量，形状为 (B, C_z, T, H_z, W_z)。
        
        返回:
            Tensor: 重构的张量，形状为 (B, C, T, H, W)。
        """
        # 获取时间维度长度
        T = z.shape[2]

        # 重排张量形状，从 (B, C_z, T, H_z, W_z) 变为 (B*T, C_z, H_z, W_z)
        z = rearrange(z, "b c t h w -> (b t) c h w")
        # 对潜在向量进行逆缩放和平移
        z = z / self.scale_factor + self.shift_factor

        # 通过解码器获取重构的张量
        x = self.decoder(z)

        # 重排回原始时间维度，从 (B*T, C, H, W) 变为 (B, C, T, H, W)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=T)
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, DiagonalGaussianDistribution, Tensor]:
        """
        前向传播方法，实现编码和解码过程。
        
        参数:
            x (Tensor): 输入张量，形状为 (B, C, T, H, W)。
        
        返回:
            Tuple[Tensor, DiagonalGaussianDistribution, Tensor]: 
                - 重构的张量 x_rec，形状为 (B, C, T, H, W)。
                - 潜在分布参数，DiagonalGaussianDistribution 对象。
                - 潜在向量 z，形状为 (B, C_z, T, H_z, W_z)。
        """
        # 编码过程
        x.shape[2]
        z, posterior = self.encode_(x)

        # 解码过程
        x_rec = self.decode(z)

        return x_rec, posterior, z

    def get_last_layer(self):
        """
        获取解码器最后一层的权重。
        
        返回:
            Tensor: 解码器最后一层的权重。
        """
        return self.decoder.conv_out.weight


@MODELS.register_module("autoencoder_2d")
def AutoEncoderFlux(
    from_pretrained: str,
    cache_dir=None,
    resolution=256,
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
    device_map: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoEncoder:
    """
    创建并加载 AutoEncoder 模型。
    
    参数:
        from_pretrained (str): 预训练模型的路径或标识符。
        cache_dir (str, optional): 缓存目录，用于存储下载的预训练模型和缓存数据。默认为 None。
        resolution (int, optional): 输入图像的分辨率。默认为 256。
        in_channels (int, optional): 输入图像的通道数。默认为 3（RGB）。
        ch (int, optional): 初始隐藏层的通道数。默认为 128。
        out_ch (int, optional): 输出图像的通道数。默认为 3（RGB）。
        ch_mult (list[int], optional): 通道数乘数列表，用于在每个下采样阶段增加通道数。默认为 [1, 2, 4, 4]。
        num_res_blocks (int, optional): 每个分辨率级别中残差块的数目。默认为 2。
        z_channels (int, optional): 潜在空间（z 空间）的通道数。默认为 16。
        scale_factor (float, optional): 缩放因子，用于调整潜在空间的大小。默认为 0.3611。
        shift_factor (float, optional): 平移因子，用于调整潜在空间的位置。默认为 0.1159。
        device_map (str | torch.device, optional): 设备映射，指定模型加载的设备。默认为 "cuda"。
        torch_dtype (torch.dtype, optional): 模型参数的数据类型。默认为 torch.bfloat16。
    
    返回:
        AutoEncoder: 加载并配置好的 AutoEncoder 模型。
    """
    # 创建配置对象
    config = AutoEncoderConfig(
        from_pretrained=from_pretrained,
        cache_dir=cache_dir,
        resolution=resolution,
        in_channels=in_channels,
        ch=ch,
        out_ch=out_ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        z_channels=z_channels,
        scale_factor=scale_factor,
        shift_factor=shift_factor,
    )
    with torch.device(device_map):
        # 创建 AutoEncoder 模型并移动到指定的数据类型
        model = AutoEncoder(config).to(torch_dtype)

    # 如果提供了预训练模型路径，则加载预训练模型
    if from_pretrained:
        model = load_checkpoint(model, from_pretrained, cache_dir=cache_dir, device_map=device_map)
    return model
