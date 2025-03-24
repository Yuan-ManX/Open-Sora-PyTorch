from colossalai.shardformer import ShardConfig, ShardFormer
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from acceleration.shardformer.policy.t5_encoder import T5EncoderPolicy
from registry import MODELS


@MODELS.register_module("text_embedder")
class HFEmbedder(nn.Module):
    """
    HFEmbedder 类用于从预训练的 Hugging Face 模型中提取文本嵌入。
    
    参数:
        from_pretrained (str): 预训练模型的名称或路径，例如 "openai/clip-vit-base-patch32" 或 "t5-base"。
        max_length (int): 模型处理的最大序列长度。
        shardformer (bool, optional): 是否使用 Shardformer 优化 T5 模型。默认为 False。
        **hf_kwargs: 其他传递给 Hugging Face 模型加载方法的关键词参数。
    """
    def __init__(self, from_pretrained: str, max_length: int, shardformer: bool = False, **hf_kwargs):
        super().__init__()
        # 判断是否使用 CLIP 模型
        self.is_clip = "openai" in from_pretrained
        # 设置最大序列长度
        self.max_length = max_length
        # 根据模型类型设置输出键
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            # 加载 CLIP 的分词器和文本模型
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(from_pretrained, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(from_pretrained, **hf_kwargs)
            # CLIP 不支持 Shardformer
            assert not shardformer, "Shardformer is not supported for CLIP"
        else:
            # 加载 T5 的分词器，使用 legacy=True 以确保兼容性
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                from_pretrained, max_length=max_length, legacy=True
            )
            # 加载 T5 的编码器模型
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(from_pretrained, **hf_kwargs)
            # 如果启用 Shardformer，则优化 T5 模型
            if shardformer:
                self.hf_module = shardformer_t5(self.hf_module)

        # 将模型设置为评估模式，并冻结参数以防止梯度更新
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str], added_tokens: int = 0, seq_align: int = 1) -> Tensor:
        """
        前向传播方法，用于获取文本的嵌入表示。
        
        参数:
            text (list[str]): 输入的文本列表。
            added_tokens (int, optional): 额外添加的标记数量。默认为 0。
            seq_align (int, optional): 序列对齐的步长。默认为 1。
        
        返回:
            torch.Tensor: 文本的嵌入表示。
        """
        # 使用分词器对文本进行编码
        batch_encoding = self.tokenizer(
            text,
            truncation=True,  # 截断超过最大长度的序列
            max_length=self.max_length,
            return_length=False,  # 不返回序列长度
            return_overflowing_tokens=False,  # 不返回溢出的标记
            padding="max_length",  # 对序列进行填充，使其长度一致
            return_tensors="pt",
        )

        # 获取序列长度
        seq_len = batch_encoding["input_ids"].shape[1]
        if (added_tokens + seq_len) % seq_align != 0:
            num_pad_tokens = seq_align - (added_tokens + seq_len) % seq_align
            # 使用填充标记进行填充
            batch_encoding["input_ids"] = nn.functional.pad(
                batch_encoding["input_ids"], (0, num_pad_tokens), value=self.tokenizer.pad_token_id
            )

        # 将输入张量移动到模型所在的设备（CPU 或 GPU）
        # 前向传播，获取模型输出
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )

        # 返回指定输出键对应的嵌入
        return outputs[self.output_key]


def shardformer_t5(t5: T5EncoderModel) -> T5EncoderModel:
    """
    使用 Shardformer 优化 T5 模型。
    
    参数:
        t5 (T5Model): 需要优化的 T5 模型。
    
    返回:
        T5Model: 优化后的 T5 模型。
    """
    # 获取模型权重的数据类型
    dtype = t5.shared.weight.dtype

    # 配置 Shardformer 的参数
    shard_config = ShardConfig(
        enable_tensor_parallelism=False,  # 不启用张量并行
        enable_jit_fused=True,  # 启用 JIT 融合
    )

    # 初始化 ShardFormer 对象
    shard_former = ShardFormer(shard_config=shard_config)

    # 使用 Shardformer 优化模型，应用 T5EncoderPolicy 策略
    optim_model, _ = shard_former.optimize(t5, policy=T5EncoderPolicy())

    # 将优化后的模型转换为指定的数据类型，并设置为评估模式，冻结参数
    optim_model = optim_model.to(dtype).eval().requires_grad_(False)
    return optim_model
