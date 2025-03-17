import math
from dataclasses import dataclass
import torch
from einops import rearrange
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from torch import Tensor, nn

from core import attention, liger_rope, rope


class EmbedND(nn.Module):
    """
    ROPE位置编码嵌入层。

    参数:
    - dim (int): 嵌入向量的维度。
    - theta (int): 控制位置编码频率的参数。
    - axes_dim (List[int]): 每个轴的维度列表。
    """
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        前向传播方法，计算ROPE位置编码。

        参数:
        - ids (Tensor): 输入张量，形状为 (..., n_axes)。

        返回:
        - Tensor: 位置编码后的张量，形状为 (..., n_axes, dim)。
        """
        # 获取轴的数量
        n_axes = ids.shape[-1]
        # 对每个轴应用ROPE位置编码，并沿最后一个维度拼接
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        # 在第二个维度上添加一个维度
        return emb.unsqueeze(1)


class LigerEmbedND(nn.Module):
    """
    Liger ROPE位置编码嵌入层。

    参数:
    - dim (int): 嵌入向量的维度。
    - theta (int): 控制位置编码频率的参数。
    - axes_dim (List[int]): 每个轴的维度列表。
    """
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        前向传播方法，计算Liger ROPE位置编码的余弦和正弦部分。

        参数:
        - ids (Tensor): 输入张量，形状为 (..., n_axes)。

        返回:
        - Tuple[Tensor, Tensor]: 余弦和正弦位置编码张量，形状均为 (..., n_axes, dim)。
        """
        # 获取轴的数量
        n_axes = ids.shape[-1]
        cos_list = []
        sin_list = []
        for i in range(n_axes):
            # 对每个轴应用Liger ROPE位置编码
            cos, sin = liger_rope(ids[..., i], self.axes_dim[i], self.theta)
            cos_list.append(cos)
            sin_list.append(sin)
        # 将所有轴的余弦编码拼接起来，并在第二个维度上重复一次
        cos_emb = torch.cat(cos_list, dim=-1).repeat(1, 1, 2).contiguous()
        # 将所有轴的正弦编码拼接起来，并在第二个维度上重复一次
        sin_emb = torch.cat(sin_list, dim=-1).repeat(1, 1, 2).contiguous()

        # 返回余弦和正弦编码张量
        return (cos_emb, sin_emb)


# 使用torch.compile进行编译优化，模式为“最大自动调优且不使用CUDA图”，并启用动态形状
@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    创建正弦时间步嵌入。

    参数:
    - t (Tensor): 一个1维张量，包含N个索引，每个批次元素一个。这些可能是分数。
    - dim (int): 输出的维度。
    - max_period (int, 可选): 控制嵌入的最小频率，默认为10000。
    - time_factor (float, 可选): 时间因子，用于缩放时间步，默认为1000.0。

    返回:
    - Tensor: 位置嵌入张量，形状为 (N, D)。
    """
    # 缩放时间步
    t = time_factor * t
    # 计算一半的维度
    half = dim // 2
    # 计算频率，使用指数函数控制频率范围
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    # 计算角度
    args = t[:, None].float() * freqs[None]
    # 计算正弦和余弦，并拼接起来
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        # 如果维度是奇数，则在末尾添加一个零张量以匹配维度
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        # 如果输入是浮点数，则将嵌入转换为相同的类型
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    """
    MLP嵌入器，用于处理输入向量。

    参数:
    - in_dim (int): 输入维度。
    - hidden_dim (int): 隐藏层维度。
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        # 定义输入线性层
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        # 定义SiLU激活函数
        self.silu = nn.SiLU()
        # 定义输出线性层
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 输出张量。
        """
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    """
    RMS归一化层。

    参数:
    - dim (int): 输入的维度。
    """
    def __init__(self, dim: int):
        super().__init__()
        # 定义缩放因子参数
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        """
        前向传播方法。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 归一化后的张量。
        """
        # 保存输入张量的数据类型
        x_dtype = x.dtype
        # 将输入转换为浮点数
        x = x.float()
        # 计算RMS值，添加小常数防止除零
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        # 应用归一化并缩放
        return (x * rrms).to(dtype=x_dtype) * self.scale


class FusedRMSNorm(RMSNorm):
    """
    融合的RMS归一化层，使用LigerRMSNormFunction进行加速。

    参数:
    - dim (int): 输入的维度。
    """
    def forward(self, x: Tensor):
        """
        前向传播方法，使用LigerRMSNormFunction进行归一化。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 归一化后的张量。
        """
        return LigerRMSNormFunction.apply(
            x,
            self.scale,
            1e-6,
            0.0,
            "llama",
            False,
        )


class QKNorm(torch.nn.Module):
    """
    QK归一化层，对查询（Q）和键（K）进行归一化处理。

    参数:
        dim (int): 输入的维度。
    """
    def __init__(self, dim: int):
        super().__init__()
        # 使用融合的RMS归一化层对Q和K进行归一化
        self.query_norm = FusedRMSNorm(dim)
        self.key_norm = FusedRMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        前向传播方法，对Q和K进行归一化处理。

        参数:
            q (Tensor): 查询张量。
            k (Tensor): 键张量。
            v (Tensor): 值张量。

        返回:
            Tuple[Tensor, Tensor]: 归一化后的Q和K张量。
        """
        # 对Q进行归一化
        q = self.query_norm(q)
        # 对K进行归一化
        k = self.key_norm(k)
        # 返回归一化后的Q和K，并确保它们的数据类型与V相同
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    """
    自注意力机制模块。

    参数:
        dim (int): 输入的维度。
        num_heads (int, 可选): 多头注意力的头数，默认为8。
        qkv_bias (bool, 可选): 在Q、K、V投影中是否使用偏置，默认为False。
        fused_qkv (bool, 可选): 是否使用融合的Q、K、V投影，默认为True。
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, fused_qkv: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        # 每个头的维度
        head_dim = dim // num_heads

        if fused_qkv:
            # 如果使用融合的Q、K、V投影，则使用一个线性层输出Q、K、V的拼接
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            # 否则，分别使用三个线性层进行Q、K、V的投影
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # 使用QK归一化层对Q和K进行归一化
        self.norm = QKNorm(head_dim)
        # 输出投影层，将注意力输出投影回原始维度
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        """
        前向传播方法，计算自注意力机制。

        参数:
            x (Tensor): 输入张量。
            pe (Tensor): 位置编码张量。

        返回:
            Tensor: 注意力机制的输出张量。
        """
        if self.fused_qkv:
            # 使用融合的Q、K、V投影
            qkv = self.qkv(x)
            # 重塑张量以分离Q、K、V
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        else:
            # 分别进行Q、K、V的投影
            q = rearrange(self.q_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
            k = rearrange(self.k_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
            v = rearrange(self.v_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
        # 对Q和K进行归一化处理
        q, k = self.norm(q, k, v)
        if not self.fused_qkv:
            # 如果不使用融合的Q、K、V投影，则重塑Q、K、V的维度
            q = rearrange(q, "B L H D -> B H L D")
            k = rearrange(k, "B L H D -> B H L D")
            v = rearrange(v, "B L H D -> B H L D")
        # 计算注意力机制
        x = attention(q, k, v, pe=pe)
        # 通过输出投影层
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    """
    调制输出数据类，用于存储调制的移位、缩放和门控值。

    参数:
        shift (Tensor): 移位张量。
        scale (Tensor): 缩放张量。
        gate (Tensor): 门控张量。
    """
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """
    调制模块，用于对输入向量进行调制处理。

    参数:
        dim (int): 输入向量的维度。
        double (bool): 是否进行双重调制。
    """
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        # 根据是否进行双重调制，设置乘数因子
        self.multiplier = 6 if double else 3
        # 定义线性层，将输入向量映射到调制参数
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        """
        前向传播方法，对输入向量进行调制处理。

        参数:
            vec (Tensor): 输入向量。

        返回:
            Tuple[ModulationOut, Optional[ModulationOut]]: 调制输出，包括主调制和可选的次调制。
        """
        # 应用SiLU激活函数后，通过线性层得到调制参数
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        # 返回主调制输出和次调制输出（如果进行双重调制）
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlockProcessor:
    """
    双流块处理器，用于处理双流块的前向传播。

    参数:
        attn (nn.Module): 双流块模块。
        img (Tensor): 输入图像张量。
        txt (Tensor): 输入文本张量。
        vec (Tensor): 输入向量张量。
        pe (Tensor): 位置编码张量。
    """
    def __call__(self, attn: nn.Module, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        # attn 是双流块模块；
        # 分别处理图像和文本，同时两者都受文本向量的影响

        # 向量将与图像潜在表示和文本上下文交互
        img_mod1, img_mod2 = attn.img_mod(vec)  # 获取每个调制的移位、缩放和门控
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # 准备图像进行注意力计算
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift

        if attn.img_attn.fused_qkv:
            # 如果使用融合的Q、K、V投影，则进行Q、K、V的投影并重塑张量
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            # 否则，分别进行Q、K、V的投影并重塑张量
            img_q = rearrange(attn.img_attn.q_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_k = rearrange(attn.img_attn.k_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_v = rearrange(attn.img_attn.v_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)

        # 对Q和K进行归一化处理
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)  # RMSNorm for QK Norm as in SD3 paper
        if not attn.img_attn.fused_qkv:
            # 如果不使用融合的Q、K、V投影，则重塑Q、K、V的维度
            img_q = rearrange(img_q, "B L H D -> B H L D")
            img_k = rearrange(img_k, "B L H D -> B H L D")
            img_v = rearrange(img_v, "B L H D -> B H L D")

        # 准备文本进行注意力计算
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if attn.txt_attn.fused_qkv:
            # 如果使用融合的Q、K、V投影，则进行Q、K、V的投影并重塑张量
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            # 否则，分别进行Q、K、V的投影并重塑张量
            txt_q = rearrange(attn.txt_attn.q_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_k = rearrange(attn.txt_attn.k_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_v = rearrange(attn.txt_attn.v_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
        # 对Q和K进行归一化处理
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)
        if not attn.txt_attn.fused_qkv:
            # 如果不使用融合的Q、K、V投影，则重塑Q、K、V的维度
            txt_q = rearrange(txt_q, "B L H D -> B H L D")
            txt_k = rearrange(txt_k, "B L H D -> B H L D")
            txt_v = rearrange(txt_v, "B L H D -> B H L D")

        # 运行实际的注意力计算，图像和文本注意力通过拼接不同的注意力头一起计算
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        # 分离文本和图像的注意力输出
        txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2] :]

        # 计算图像块
        # 应用注意力输出并更新图像张量
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        # 应用MLP并更新图像张量
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # 计算文本块
        # 应用注意力输出并更新文本张量
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        # 应用MLP并更新文本张量
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class DoubleStreamBlock(nn.Module):
    """
    双流块类，处理图像和文本输入，并结合向量输入进行调制和注意力计算。

    参数:
        hidden_size (int): 隐藏层大小。
        num_heads (int): 多头注意力的头数。
        mlp_ratio (float): MLP层的隐藏层大小与输入大小的比率。
        qkv_bias (bool, 可选): 在Q、K、V投影中是否使用偏置，默认为False。
        fused_qkv (bool, 可选): 是否使用融合的Q、K、V投影，默认为True。
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        fused_qkv: bool = True,
    ):
        super().__init__()
        # 计算MLP层的隐藏层大小
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        # 计算每个头的维度
        self.head_dim = hidden_size // num_heads

        # 图像流部分
        # 图像调制模块
        self.img_mod = Modulation(hidden_size, double=True)
        # 图像LayerNorm层
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 图像自注意力模块
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, fused_qkv=fused_qkv)

        # 图像LayerNorm层
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 图像MLP模块
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # 文本流部分
        # 文本调制模块
        self.txt_mod = Modulation(hidden_size, double=True)
        # 文本LayerNorm层
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 文本自注意力模块
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, fused_qkv=fused_qkv)

        # 文本LayerNorm层
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 文本MLP模块
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # 处理器部分
        # 创建双流块处理器实例
        processor = DoubleStreamBlockProcessor()
        # 设置处理器
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        """
        设置处理器。

        参数:
            processor (DoubleStreamBlockProcessor): 双流块处理器实例。
        """
        self.processor = processor

    def get_processor(self):
        """
        获取处理器。

        返回:
            DoubleStreamBlockProcessor: 双流块处理器实例。
        """
        return self.processor

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        """
        前向传播方法，处理图像和文本输入，并结合向量输入进行调制和注意力计算。

        参数:
            img (Tensor): 输入图像张量。
            txt (Tensor): 输入文本张量。
            vec (Tensor): 输入向量张量。
            pe (Tensor): 位置编码张量。

        返回:
            Tuple[Tensor, Tensor]: 处理后的图像和文本张量。
        """
        return self.processor(self, img, txt, vec, pe)


class SingleStreamBlockProcessor:
    """
    单流块处理器，用于处理单流块的前向传播。

    参数:
        attn (nn.Module): 单流块模块。
        x (Tensor): 输入张量。
        vec (Tensor): 输入向量张量。
        pe (Tensor): 位置编码张量。
    """
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        """
        前向传播方法，处理输入张量，并结合向量输入进行调制和注意力计算。

        参数:
            attn (nn.Module): 单流块模块。
            x (Tensor): 输入张量。
            vec (Tensor): 输入向量张量。
            pe (Tensor): 位置编码张量。

        返回:
            Tensor: 处理后的输出张量。
        """
        # 获取调制输出
        mod, _ = attn.modulation(vec)
        # 应用调制
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        if attn.fused_qkv:
            # 如果使用融合的Q、K、V投影，则进行Q、K、V的投影并重塑张量
            qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        else:
            # 否则，分别进行Q、K、V的投影并重塑张量
            q = rearrange(attn.q_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            k = rearrange(attn.k_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            v, mlp = torch.split(attn.v_mlp(x_mod), [attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            v = rearrange(v, "B L (H D) -> B L H D", H=attn.num_heads)

        # 对Q和K进行归一化处理
        q, k = attn.norm(q, k, v)
        if not attn.fused_qkv:
            # 如果不使用融合的Q、K、V投影，则重塑Q、K、V的维度
            q = rearrange(q, "B L H D -> B H L D")
            k = rearrange(k, "B L H D -> B H L D")
            v = rearrange(v, "B L H D -> B H L D")

        # 计算注意力机制
        attn_1 = attention(q, k, v, pe=pe)

        # 在MLP流中计算激活，并再次拼接并运行第二个线性层
        # 应用第二个线性层
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        # 应用门控并更新输出
        output = x + mod.gate * output
        return output


class SingleStreamBlock(nn.Module):
    """
    单流块类，类似于DiT块，但具有并行的线性层和调整后的调制接口。

    参数:
        hidden_size (int): 隐藏层大小。
        num_heads (int): 多头注意力的头数。
        mlp_ratio (float, 可选): MLP层的隐藏层大小与输入大小的比率，默认为4.0。
        qk_scale (float | None, 可选): Q和K的缩放因子，如果为None，则使用head_dim的-0.5次方，默认为None。
        fused_qkv (bool, 可选): 是否使用融合的Q、K、V投影，默认为True。
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        fused_qkv: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        # 计算每个头的维度
        self.head_dim = hidden_size // num_heads
        # 计算Q和K的缩放因子
        self.scale = qk_scale or self.head_dim**-0.5
        self.fused_qkv = fused_qkv

        # 计算MLP层的隐藏层大小
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if fused_qkv:
            # 如果使用融合的Q、K、V投影，则定义一个线性层输出Q、K、V和MLP输入的拼接
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        else:
            # 否则，分别定义Q、K、V和MLP输入的线性层
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_mlp = nn.Linear(hidden_size, hidden_size + self.mlp_hidden_dim)

        # 定义输出线性层和MLP输出线性层
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        # 定义QK归一化层
        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        # 定义预归一化层
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # 定义激活函数
        self.mlp_act = nn.GELU(approximate="tanh")
        # 定义调制模块
        self.modulation = Modulation(hidden_size, double=False)

        # 创建单流块处理器实例
        processor = SingleStreamBlockProcessor()
        # 设置处理器
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        """
        设置处理器。

        参数:
            processor (SingleStreamBlockProcessor): 单流块处理器实例。
        """
        self.processor = processor

    def get_processor(self):
        """
        获取处理器。

        返回:
            SingleStreamBlockProcessor: 单流块处理器实例。
        """
        return self.processor

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        """
        前向传播方法，处理输入张量，并结合向量输入进行调制和注意力计算。

        参数:
            x (Tensor): 输入张量。
            vec (Tensor): 输入向量张量。
            pe (Tensor): 位置编码张量。

        返回:
            Tensor: 处理后的输出张量。
        """
        return self.processor(self, x, vec, pe)


class LastLayer(nn.Module):
    """
    最后一层类，用于处理输出张量。

    参数:
        hidden_size (int): 隐藏层大小。
        patch_size (int): 补丁大小。
        out_channels (int): 输出通道数。
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        # 定义最终归一化层
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 定义线性层
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # 定义自适应归一化调制模块
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        """
        前向传播方法，处理输入张量，并结合向量输入进行自适应归一化调制。

        参数:
            x (Tensor): 输入张量。
            vec (Tensor): 输入向量张量。

        返回:
            Tensor: 输出张量。
        """
        # 获取移位和缩放因子
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        # 应用自适应归一化调制
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        # 通过线性层
        x = self.linear(x)
        return x
