from dataclasses import dataclass
import torch
from torch import Tensor, nn

from acceleration.checkpoint import auto_grad_checkpoint
from layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    LigerEmbedND,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from registry import MODELS
from utils.ckpt import load_checkpoint


# 数据类，用于配置MMDiT模型
@dataclass
class MMDiTConfig:
    """
    MMDiT模型配置类，用于初始化MMDiTModel。

    参数:
    - model_type (str): 模型类型，默认为"MMDiT"。
    - from_pretrained (str): 预训练模型的路径或标识符。
    - cache_dir (str): 缓存目录，用于存储下载的预训练模型。
    - in_channels (int): 输入通道数，表示输入数据的维度。
    - vec_in_dim (int): 向量输入的维度，用于处理向量输入数据。
    - context_in_dim (int): 上下文输入的维度，用于处理上下文信息。
    - hidden_size (int): 隐藏层的大小，决定了模型内部表示的维度。
    - mlp_ratio (float): MLP层中隐藏层大小的比例，通常用于确定MLP的宽度。
    - num_heads (int): 多头注意力机制中的头数。
    - depth (int): 双流块（DoubleStreamBlock）的层数。
    - depth_single_blocks (int): 单流块（SingleStreamBlock）的层数。
    - axes_dim (List[int]): 位置编码中每个轴的维度，必须与pe_dim匹配。
    - theta (int): 位置编码中的参数，用于控制位置编码的频率。
    - qkv_bias (bool): 在QKV（查询、键、值）投影中是否使用偏置。
    - guidance_embed (bool): 是否使用引导嵌入。
    - cond_embed (bool, 可选): 是否使用条件嵌入，默认为False。
    - fused_qkv (bool, 可选): 是否融合QKV操作，默认为True。
    - grad_ckpt_settings (Tuple[int, int] | None, 可选): 梯度检查点设置，用于节省显存，默认为None。
    - use_liger_rope (bool, 可选): 是否使用Liger ROPE位置编码，默认为False。
    - patch_size (int, 可选): patch大小，用于分割输入图像，默认为2。
    """
    model_type = "MMDiT"
    from_pretrained: str
    cache_dir: str
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    cond_embed: bool = False
    fused_qkv: bool = True
    grad_ckpt_settings: tuple[int, int] | None = None
    use_liger_rope: bool = False
    patch_size: int = 2

    def get(self, attribute_name, default=None):
        """
        获取配置属性值。

        参数:
        - attribute_name (str): 属性名称。
        - default: 默认值，如果属性不存在则返回该值。

        返回:
        - 属性值或默认值。
        """
        return getattr(self, attribute_name, default)

    def __contains__(self, attribute_name):
        """
        检查配置中是否存在某个属性。

        参数:
        - attribute_name (str): 属性名称。

        返回:
        - 布尔值，表示属性是否存在。
        """
        return hasattr(self, attribute_name)


class MMDiTModel(nn.Module):
    """
    MMDiT模型类，定义了模型的结构和前向传播过程。

    参数:
    - config (MMDiTConfig): 配置对象，包含模型的所有配置参数。
    """
    config_class = MMDiTConfig

    def __init__(self, config: MMDiTConfig):
        super().__init__()

        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        self.patch_size = config.patch_size

        # 检查隐藏层大小是否可以被头数整除
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )

        pe_dim = config.hidden_size // config.num_heads
        # 检查axes_dim的总和是否与pe_dim匹配
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.axes_dim} but expected positional dim {pe_dim}"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # 根据配置选择位置编码器
        pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
        self.pe_embedder = pe_embedder_cls(
            dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim
        )

        # 定义输入线性层，将输入通道数映射到隐藏层大小
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        # 定义时间嵌入的MLP嵌入器
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        # 定义向量输入的MLP嵌入器
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)
        # 根据配置决定是否使用引导嵌入
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if config.guidance_embed
            else nn.Identity()
        )
        # 根据配置决定是否使用条件嵌入
        self.cond_in = (
            nn.Linear(
                self.in_channels + self.patch_size**2, self.hidden_size, bias=True
            )
            if config.cond_embed
            else nn.Identity()
        )
        # 定义文本输入的线性层
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)

        # 定义双流块列表
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    fused_qkv=config.fused_qkv,
                )
                for _ in range(config.depth)
            ]
        )

        # 定义单流块列表
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    fused_qkv=config.fused_qkv,
                )
                for _ in range(config.depth_single_blocks)
            ]
        )

        # 定义最后一层，用于输出最终结果
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        # 初始化模型权重
        self.initialize_weights()

        # 根据梯度检查点设置决定前向传播方法
        if self.config.grad_ckpt_settings:
            self.forward = self.forward_selective_ckpt
        else:
            self.forward = self.forward_ckpt
        self._input_requires_grad = False

    def initialize_weights(self):
        """
        初始化模型权重的方法。
        """
        # 如果使用条件嵌入，则将条件嵌入层的权重和偏置初始化为零
        if self.config.cond_embed:
            nn.init.zeros_(self.cond_in.weight)
            nn.init.zeros_(self.cond_in.bias)

    def prepare_block_inputs(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,  # t5 encoded vec
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,  # clip encoded vec
        cond: Tensor = None,
        guidance: Tensor | None = None,
    ):
        """
        处理输入数据，得到以下处理后的张量：
            img: 投影后的噪声图像潜在表示，
            txt: 文本上下文（来自T5），
            vec: clip 编码的向量，
            pe: 拼接后的 img 和 txt 的位置嵌入。

        参数:
        - img (Tensor): 输入图像张量。
        - img_ids (Tensor): 图像标识符张量。
        - txt (Tensor): 文本编码向量（来自T5）。
        - txt_ids (Tensor): 文本标识符张量。
        - timesteps (Tensor): 时间步张量。
        - y_vec (Tensor): clip 编码的向量。
        - cond (Tensor, 可选): 条件输入，默认为None。
        - guidance (Tensor | float, 可选): 引导强度，默认为None。

        返回:
        - Tuple[Tensor, Tensor, Tensor, Tensor]: 处理后的 img, txt, vec, pe 张量。
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # 对图像序列进行处理
        img = self.img_in(img)
        if self.config.cond_embed:
            if cond is None:
                raise ValueError("Didn't get conditional input for conditional model.")
            img = img + self.cond_in(cond)

        # 生成时间步嵌入向量
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y_vec)

        # 处理文本输入
        txt = self.txt_in(txt)

        # 拼接文本和图像标识符
        # concat: 4096 + t*h*2/4
        ids = torch.cat((txt_ids, img_ids), dim=1)
        # 生成位置嵌入
        pe = self.pe_embedder(ids)

        if self._input_requires_grad:
            # 仅对双流和单流块应用LoRA，因此只需对这些输入启用梯度
            img.requires_grad_()
            txt.requires_grad_()

        return img, txt, vec, pe

    def enable_input_require_grads(self):
        """
        启用输入梯度的要求。此方法不应手动调用。
        """
        self._input_requires_grad = True

    def forward_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,
        cond: Tensor = None,
        guidance: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """
        使用梯度检查点的前向传播方法。

        参数:
        - img (Tensor): 输入图像张量。
        - img_ids (Tensor): 图像标识符张量。
        - txt (Tensor): 文本编码向量（来自T5）。
        - txt_ids (Tensor): 文本标识符张量。
        - timesteps (Tensor): 时间步张量。
        - y_vec (Tensor): clip 编码的向量。
        - cond (Tensor, 可选): 条件输入，默认为None。
        - guidance (Tensor | float, 可选): 引导强度，默认为None。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 输出张量。
        """
        img, txt, vec, pe = self.prepare_block_inputs(
            img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance
        )

        # 对双流块应用自动梯度检查点
        for block in self.double_blocks:
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)

        # 拼接文本和图像张量
        img = torch.cat((txt, img), 1)
        # 对单流块应用自动梯度检查点
        for block in self.single_blocks:
            img = auto_grad_checkpoint(block, img, vec, pe)
        # 去除文本部分，只保留图像部分
        img = img[:, txt.shape[1] :, ...]

        # 通过最后一层处理输出
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward_selective_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,
        cond: Tensor = None,
        guidance: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """
        选择性使用梯度检查点的前向传播方法。

        参数:
        - img (Tensor): 输入图像张量。
        - img_ids (Tensor): 图像标识符张量。
        - txt (Tensor): 文本编码向量（来自T5）。
        - txt_ids (Tensor): 文本标识符张量。
        - timesteps (Tensor): 时间步张量。
        - y_vec (Tensor): clip 编码的向量。
        - cond (Tensor, 可选): 条件输入，默认为None。
        - guidance (Tensor | float, 可选): 引导强度，默认为None。
        - **kwargs: 其他关键字参数。

        返回:
        - Tensor: 输出张量。
        """
        img, txt, vec, pe = self.prepare_block_inputs(
            img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance
        )

        # 根据配置中的梯度检查点设置，获取双流块的检查点深度
        ckpt_depth_double = self.config.grad_ckpt_settings[0]
        for block in self.double_blocks[:ckpt_depth_double]:
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)

        for block in self.double_blocks[ckpt_depth_double:]:
            img, txt = block(img, txt, vec, pe)

        # 拼接文本和图像张量
        ckpt_depth_single = self.config.grad_ckpt_settings[1]
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks[:ckpt_depth_single]:
            img = auto_grad_checkpoint(block, img, vec, pe)
        for block in self.single_blocks[ckpt_depth_single:]:
            img = block(img, vec, pe)

        # 去除文本部分，只保留图像部分
        img = img[:, txt.shape[1] :, ...]

        # 通过最后一层处理输出
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


@MODELS.register_module("flux")
def Flux(
    cache_dir: str = None,
    from_pretrained: str = None,
    device_map: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    strict_load: bool = False,
    **kwargs,
) -> MMDiTModel:
    """
    工厂函数，用于创建和加载 MMDiTModel 模型。

    参数:
    - cache_dir (str, 可选): 缓存目录，用于存储下载的预训练模型。
    - from_pretrained (str, 可选): 预训练模型的路径或标识符。
    - device_map (str | torch.device, 可选): 设备映射，指定模型加载到哪个设备，默认为"cuda"。
    - torch_dtype (torch.dtype, 可选): 模型的数据类型，默认为 torch.bfloat16。
    - strict_load (bool, 可选): 是否严格加载模型参数，默认为False。
    - **kwargs: 其他关键字参数。

    返回:
    - MMDiTModel: 初始化并加载了预训练参数的 MMDiTModel 模型。
    """
    # 创建配置对象
    config = MMDiTConfig(
        from_pretrained=from_pretrained,
        cache_dir=cache_dir,
        **kwargs,
    )

    # 判断是否使用低精度初始化
    low_precision_init = from_pretrained is not None and len(from_pretrained) > 0
    if low_precision_init:
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
    
    # 设置设备并初始化模型
    with torch.device(device_map):
        model = MMDiTModel(config)
    if low_precision_init:
        torch.set_default_dtype(default_dtype)
    else:
        model = model.to(torch_dtype)

    # 如果提供了预训练路径，则加载预训练模型
    if from_pretrained:
        model = load_checkpoint(
            model,
            from_pretrained,
            cache_dir=cache_dir,
            device_map=device_map,
            strict=strict_load,
        )
        
    return model
