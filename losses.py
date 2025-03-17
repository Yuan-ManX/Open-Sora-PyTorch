import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from lpips import LPIPS


def hinge_d_loss(logits_real, logits_fake):
    """
    计算Hinge对抗性损失。

    参数:
        logits_real (Tensor): 判别器对真实样本的输出对数。
        logits_fake (Tensor): 判别器对生成样本的输出对数。

    返回:
        Tensor: Hinge损失值。
    """
    # 计算真实样本的损失：ReLU(1 - logits_real)的均值
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    # 计算生成样本的损失：ReLU(1 + logits_fake)的均值
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    # 总损失为真实样本损失和生成样本损失的平均值
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    """
    计算Vanilla对抗性损失。

    参数:
        logits_real (Tensor): 判别器对真实样本的输出对数。
        logits_fake (Tensor): 判别器对生成样本的输出对数。

    返回:
        Tensor: Vanilla对抗性损失值。
    """
    # 总损失为真实样本损失和生成样本损失的平均值
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def wgan_gp_loss(logits_real, logits_fake):
    """
    计算WGAN-GP对抗性损失。

    参数:
        logits_real (Tensor): 判别器对真实样本的输出对数。
        logits_fake (Tensor): 判别器对生成样本的输出对数。

    返回:
        Tensor: WGAN-GP损失值。
    """
    # 计算WGAN-GP损失：-logits_real的均值 + logits_fake的均值
    d_loss = 0.5 * (-logits_real.mean() + logits_fake.mean())
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    """
    根据全局步骤调整权重。

    参数:
        weight (float): 原始权重。
        global_step (int): 全局训练步骤。
        threshold (int, 可选): 阈值，默认为0。
        value (float, 可选): 当全局步骤小于阈值时，权重被设置为该值，默认为0.0。

    返回:
        float: 调整后的权重。
    """
    if global_step < threshold:
        # 如果全局步骤小于阈值，则将权重设置为指定值
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    """
    测量聚类的困惑度。

    参数:
        predicted_indices (Tensor): 预测的索引张量。
        n_embed (int): 嵌入的总数。

    返回:
        Tuple[Tensor, Tensor]: 困惑度和使用的聚类数量。
    """
    # 将预测的索引转换为one-hot编码
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    # 计算每个聚类的平均概率
    avg_probs = encodings.mean(0)
    # 计算困惑度
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    # 计算使用的聚类数量
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    """
    计算L1损失。

    参数:
        x (Tensor): 第一个输入张量。
        y (Tensor): 第二个输入张量。

    返回:
        Tensor: L1损失值。
    """
    return torch.abs(x - y)


def l2(x, y):
    """
    计算L2损失。

    参数:
        x (Tensor): 第一个输入张量。
        y (Tensor): 第二个输入张量。

    返回:
        Tensor: L2损失值。
    """
    return torch.pow((x - y), 2)


def batch_mean(x):
    """
    计算批量均值。

    参数:
        x (Tensor): 输入张量。

    返回:
        Tensor: 均值张量。
    """
    return torch.sum(x) / x.shape[0]


def sigmoid_cross_entropy_with_logits(labels, logits):
    """
    计算带逻辑的Sigmoid交叉熵损失。

    参数:
        labels (Tensor): 标签张量。
        logits (Tensor): 逻辑张量。

    返回:
        Tensor: 交叉熵损失值。
    """
    # 最终的公式是: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = logits >= zeros
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))


def lecam_reg(real_pred, fake_pred, ema_real_pred, ema_fake_pred):
    """
    计算LeCam正则化损失。

    参数:
        real_pred (Tensor): 判别器对真实样本的预测。
        fake_pred (Tensor): 判别器对生成样本的预测。
        ema_real_pred (Tensor): 指数移动平均的判别器对真实样本的预测。
        ema_fake_pred (Tensor): 指数移动平均的判别器对生成样本的预测。

    返回:
        Tensor: LeCam正则化损失值。
    """
    assert real_pred.ndim == 0 and ema_fake_pred.ndim == 0
    # 计算LeCam损失
    lecam_loss = torch.mean(torch.pow(nn.ReLU()(real_pred - ema_fake_pred), 2))
    lecam_loss += torch.mean(torch.pow(nn.ReLU()(ema_real_pred - fake_pred), 2))
    return lecam_loss


def gradient_penalty_fn(images, output):
    """
    计算梯度惩罚。

    参数:
        images (Tensor): 输入图像张量。
        output (Tensor): 输出张量。

    返回:
        Tensor: 梯度惩罚值。
    """
    # 计算输出关于输入的梯度
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # 重塑梯度张量
    gradients = rearrange(gradients, "b ... -> b (...)")
    # 计算梯度惩罚
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class VAELoss(nn.Module):
    """
    VAE损失类，用于计算VAE模型的损失。

    参数:
        logvar_init (float, 可选): log方差参数的初始值，默认为0.0。
        perceptual_loss_weight (float, 可选): 感知损失的权重，默认为1.0。
        kl_loss_weight (float, 可选): KL损失的权重，默认为5e-4。
        device (str, 可选): 设备类型，默认为"cpu"。
        dtype (str, 可选): 数据类型，默认为"bf16"。
    """
    def __init__(
        self,
        logvar_init=0.0,
        perceptual_loss_weight=1.0,
        kl_loss_weight=5e-4,
        device="cpu",
        dtype="bf16",
    ):
        super().__init__()

        if type(dtype) == str:
            if dtype == "bf16":
                dtype = torch.bfloat16
            elif dtype == "fp16":
                dtype = torch.float16
            elif dtype == "fp32":
                dtype = torch.float32
            else:
                raise NotImplementedError(f"dtype: {dtype}")

        # KL损失权重
        self.kl_weight = kl_loss_weight
        # 感知损失函数
        self.perceptual_loss_fn = LPIPS().eval().to(device, dtype) # 感知损失函数通常使用fp32
        # 感知损失函数不需要梯度
        self.perceptual_loss_fn.requires_grad_(False)
        self.perceptual_loss_weight = perceptual_loss_weight
        # log方差参数，用于计算KL损失
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(
        self,
        video,
        recon_video,
        posterior,
    ) -> dict:
        """
        前向传播方法，计算VAE的损失。

        参数:
            video (Tensor): 原始视频张量，形状为 (B, C, T, H, W)。
            recon_video (Tensor): 重构的视频张量，形状为 (B, C, T, H, W)。
            posterior (Any, 可选): 后验分布对象，用于计算KL损失，默认为None。

        返回:
            Dict[str, Tensor]: 包含不同损失项的字典。
        """
        # 重塑视频张量以便于计算
        video.size(0)
        video = rearrange(video, "b c t h w -> (b t) c h w").contiguous()
        recon_video = rearrange(recon_video, "b c t h w -> (b t) c h w").contiguous()

        # 重构损失：L1损失
        recon_loss = l1(video, recon_video)

        # 感知损失
        perceptual_loss = self.perceptual_loss_fn(video, recon_video)
        # NLL损失（由重构损失和感知损失组成）
        nll_loss = recon_loss + perceptual_loss * self.perceptual_loss_weight
        # 计算 NLL 损失：NLL = (损失 / exp(logvar)) + logvar
        nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar

        # 计算批量均值
        nll_loss = batch_mean(nll_loss)
        recon_loss = batch_mean(recon_loss)
        # 计算每个样本的元素数量
        numel_elements = video.numel() // video.size(0)
        # 计算感知损失的批量均值
        perceptual_loss = batch_mean(perceptual_loss) * numel_elements

        # 计算KL损失
        if posterior is None:
            kl_loss = torch.tensor(0.0).to(video.device, video.dtype)
        else:
            # 计算KL损失
            kl_loss = posterior.kl()
            # 计算批量均值
            kl_loss = batch_mean(kl_loss)
        # 加权KL损失
        weighted_kl_loss = kl_loss * self.kl_weight

        return {
            "nll_loss": nll_loss,  # NLL损失
            "kl_loss": weighted_kl_loss,  # 加权KL损失
            "recon_loss": recon_loss,  # 重构损失
            "perceptual_loss": perceptual_loss,  # 感知损失
        }


class GeneratorLoss(nn.Module):
    """
    生成器损失类，用于计算生成器的损失。

    参数:
        gen_start (int, 可选): 训练的起始步数，默认为2001。
        disc_factor (float, 可选): 判别器因子，默认为1.0。
        disc_weight (float, 可选): 判别器权重，默认为0.5。
    """
    def __init__(self, gen_start=2001, disc_factor=1.0, disc_weight=0.5):
        super().__init__()
        self.disc_factor = disc_factor
        self.gen_start = gen_start
        self.disc_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        """
        计算自适应权重。

        参数:
            nll_loss (Tensor): NLL损失。
            g_loss (Tensor): 生成器损失。
            last_layer (Tensor): 最后一层的输出。

        返回:
            Tensor: 自适应权重。
        """
        # 计算NLL损失的梯度
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        # 计算生成器损失的梯度
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        # 计算自适应权重：||nll_grads|| / (||g_grads|| + 1e-4)
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        # 限制自适应权重的范围
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # 乘以判别器权重
        d_weight = d_weight * self.disc_weight
        return d_weight

    def forward(
        self,
        logits_fake,
        nll_loss,
        last_layer,
        global_step,
        is_training=True,
    ):
        """
        前向传播方法，计算生成器损失。

        参数:
            logits_fake (Tensor): 生成器输出的对数。
            nll_loss (Tensor): NLL损失。
            last_layer (Tensor): 最后一层的输出。
            global_step (int): 全局训练步骤。
            is_training (bool, 可选): 是否处于训练模式，默认为True。

        返回:
            Tuple[Tensor, Tensor]: 加权生成器损失和生成器损失。
        """
        # 生成器损失：-logits_fake的均值
        g_loss = -torch.mean(logits_fake)

        if self.disc_factor is not None and self.disc_factor > 0.0:
            # 如果判别器因子不为None且大于0，则计算自适应权重
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
        else:
            # 否则，权重设为1.0
            d_weight = torch.tensor(1.0)

        # 应用权重调整函数调整判别器因子
        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.gen_start)
        # 计算加权生成器损失
        weighted_gen_loss = d_weight * disc_factor * g_loss

        return weighted_gen_loss, g_loss


class DiscriminatorLoss(nn.Module):
    """
    判别器损失类，用于计算判别器的损失。

    参数:
        disc_start (int, 可选): 训练的起始步数，默认为2001。
        disc_factor (float, 可选): 判别器因子，默认为1.0。
        disc_loss_type (str, 可选): 判别器损失类型，默认为"hinge"。
    """
    def __init__(self, disc_start=2001, disc_factor=1.0, disc_loss_type="hinge"):
        super().__init__()

        assert disc_loss_type in ["hinge", "vanilla", "wgan-gp"]
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.disc_loss_type = disc_loss_type

        # 根据损失类型选择损失函数
        if self.disc_loss_type == "hinge":
            self.loss_fn = hinge_d_loss
        elif self.disc_loss_type == "vanilla":
            self.loss_fn = vanilla_d_loss
        elif self.disc_loss_type == "wgan-gp":
            self.loss_fn = wgan_gp_loss
        else:
            raise ValueError(f"Unknown GAN loss '{self.disc_loss_type}'.")

    def forward(
        self,
        real_logits,
        fake_logits,
        global_step,
    ):
        """
        前向传播方法，计算判别器损失。

        参数:
            real_logits (Tensor): 判别器对真实样本的输出对数。
            fake_logits (Tensor): 判别器对生成样本的输出对数。
            global_step (int): 全局训练步骤。

        返回:
            Tensor: 加权判别器损失。
        """
        if self.disc_factor is not None and self.disc_factor > 0.0:
            # 如果判别器因子不为None且大于0，则应用权重调整函数并计算判别器损失
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
            disc_loss = self.loss_fn(real_logits, fake_logits)
            weighted_discriminator_loss = disc_factor * disc_loss
        else:
            # 否则，判别器损失为0
            weighted_discriminator_loss = 0

        return weighted_discriminator_loss
