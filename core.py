import torch
from einops import rearrange
from flash_attn import flash_attn_func as flash_attn_func_v2
from liger_kernel.ops.rope import LigerRopeFunction
from torch import Tensor, Tuple

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3

    SUPPORT_FA3 = True
except:
    SUPPORT_FA3 = False


def flash_attn_func(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """
    使用Flash Attention计算注意力机制。

    参数:
        q (Tensor): 查询张量，形状为 (B, H, L, D)。
        k (Tensor): 键张量，形状为 (B, H, L, D)。
        v (Tensor): 值张量，形状为 (B, H, L, D)。

    返回:
        Tensor: 注意力机制的输出张量。
    """
    if SUPPORT_FA3:
        # 如果支持Flash Attention v3，则调用相应的函数
        return flash_attn_func_v3(q, k, v)[0]
    # 否则，使用Flash Attention v2
    return flash_attn_func_v2(q, k, v)


def attention(q: Tensor, k: Tensor, v: Tensor, pe) -> Tensor:
    """
    计算注意力机制，并应用旋转位置编码（ROPE）。

    参数:
        q (Tensor): 查询张量，形状为 (B, H, L, D)。
        k (Tensor): 键张量，形状为 (B, H, L, D)。
        v (Tensor): 值张量，形状为 (B, H, L, D)。
        pe (Tensor | Tuple[Tensor, Tensor]): 位置编码。可以是张量或包含余弦和正弦张量的元组。

    返回:
        Tensor: 注意力机制的输出张量。
    """
    if isinstance(pe, torch.Tensor):
        # 如果位置编码是张量，则应用旋转位置编码（ROPE）
        q, k = apply_rope(q, k, pe)
    else:
        # 如果位置编码是包含余弦和正弦张量的元组，则使用Liger Rope函数
        cos, sin = pe
        q, k = LigerRopeFunction.apply(q, k, cos, sin)
        # 如果需要与原始实现进行比较，可以取消注释以下行
        # k = reverse_rearrange_tensor(k)

    # 重塑张量以适应注意力计算
    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")

    # 使用Flash Attention函数计算注意力
    x = flash_attn_func(q, k, v)
    # 重塑输出张量以匹配原始形状
    x = rearrange(x, "B L H D -> B L (H D)")

    return x


def liger_rope(pos: Tensor, dim: int, theta: int) -> Tuple:
    """
    计算Liger ROPE位置编码的余弦和正弦部分。

    参数:
        pos (Tensor): 输入位置张量，形状为 (..., N)。
        dim (int): 嵌入向量的维度，必须是偶数。
        theta (int): 控制位置编码频率的参数。

    返回:
        Tuple[Tensor, Tensor]: 余弦和正弦位置编码张量，形状均为 (..., N, dim//2)。
    """
    assert dim % 2 == 0
    # 计算缩放因子，使用对数尺度控制频率
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    # 计算角频率
    omega = 1.0 / (theta**scale)
    # 计算余弦和正弦
    out = torch.einsum("...n,d->...nd", pos, omega)  # (b, seq, dim//2)
    cos = out.cos()
    sin = out.sin()

    return (cos, sin)


def rope(pos: Tensor, dim: int, theta: int) -> Tuple:
    """
    计算旋转位置编码（ROPE）。

    参数:
        pos (Tensor): 输入位置张量，形状为 (..., N)。
        dim (int): 嵌入向量的维度，必须是偶数。
        theta (int): 控制位置编码频率的参数。

    返回:
        Tensor: 旋转位置编码张量，形状为 (..., N, dim)。
    """
    assert dim % 2 == 0
    # 计算缩放因子，使用双精度浮点数以提高精度
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    # 计算角频率
    omega = 1.0 / (theta**scale)
    # 计算余弦和正弦，并堆叠以形成复数表示
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    # 重塑张量以匹配旋转位置编码的形状
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    # 返回浮点数类型的张量
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    应用旋转位置编码（ROPE）到查询和键张量。

    参数:
        xq (Tensor): 查询张量，形状为 (..., N, D)。
        xk (Tensor): 键张量，形状为 (..., N, D)。
        freqs_cis (Tensor): 复数频率张量，形状为 (..., N, D//2, 2)。

    返回:
        Tuple[Tensor, Tensor]: 应用了ROPE后的查询和键张量。
    """
    # 将查询和键张量转换为浮点数，并重塑为 (..., N, D//2, 1, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    # 应用ROPE到查询和键张量
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    # 重塑回原始形状，并转换回原始数据类型
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def rearrange_tensor(tensor):
    """
    根据指定的映射规则重新排列输入张量最后一个维度（D）的元素：
    将2d映射到d，将2d+1映射到D/2 + d。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 [B, H, L, D]，其中D必须是偶数。

    返回:
        torch.Tensor: 重新排列后的张量，形状与输入相同。
    """
    # 获取输入张量的维度
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")
    
    # 计算D的一半
    half_D = D // 2
    # 创建一个空索引张量，形状为 (D,)，数据类型为长整型，设备与输入张量相同
    indices = torch.empty(D, dtype=torch.long, device=tensor.device)

    # 根据映射规则填充索引：
    # 前半部分索引 [0, 1, 2, ..., D/2-1] 映射到 [0, 2, 4, ..., D-2]
    indices[:half_D] = torch.arange(0, D, 2, device=tensor.device)
    # 后半部分索引 [D/2, D/2+1, ..., D-1] 映射到 [1, 3, 5, ..., D-1]
    indices[half_D:] = torch.arange(1, D, 2, device=tensor.device)

    # 根据计算出的索引重新排列张量的最后一个维度
    return tensor.index_select(dim=-1, index=indices)


def reverse_rearrange_tensor(tensor):
    """
    根据反向映射规则恢复输入张量最后一个维度（D）的原始顺序：
    将d映射到2d，将D/2 + d映射到2d + 1。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 [B, H, L, D]，其中D必须是偶数。

    返回:
        torch.Tensor: 恢复原始顺序后的张量，形状与输入相同。
    """
    # 获取输入张量的维度
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    # 计算D的一半
    half_D = D // 2
    # 创建一个空索引张量，形状为 (D,)，数据类型为长整型，设备与输入张量相同
    reverse_indices = torch.empty(D, dtype=torch.long, device=tensor.device)

    # 根据反向映射规则填充索引：
    # 偶数索引 [0, 2, 4, ..., D-2] 映射到 [0, 1, 2, ..., D/2-1]
    reverse_indices[::2] = torch.arange(half_D, device=tensor.device)
    # 奇数索引 [1, 3, 5, ..., D-1] 映射到 [D/2, D/2+1, ..., D-1]
    reverse_indices[1::2] = torch.arange(half_D, D, device=tensor.device)

    # 根据计算出的反向索引重新排列张量的最后一个维度
    return tensor.index_select(dim=-1, index=reverse_indices)
