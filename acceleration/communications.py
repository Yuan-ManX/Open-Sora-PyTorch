import torch
import torch.distributed as dist


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    """
    执行 all-to-all 通信操作，将输入张量分散到所有进程中，然后收集结果。

    参数:
        input_ (torch.Tensor): 输入张量，需要在 scatter_dim 维度上可以被 world_size 整除。
        world_size (int): 进程组中的进程数量。
        group (dist.ProcessGroup): 通信进程组。
        scatter_dim (int): 分散操作的维度。
        gather_dim (int): 收集操作的维度。

    返回:
        torch.Tensor: 收集后的张量。
    """
    # 将输入张量在 scatter_dim 维度上分割成 world_size 份，并确保每份是连续的
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]

    # 为每个进程创建一个空的输出张量列表，形状与 input_list 中的张量相同
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

    # 执行 all-to-all 通信操作，将 input_list 中的张量发送到对应的进程，并接收来自其他进程的输入
    dist.all_to_all(output_list, input_list, group=group)

    # 将接收到的输出张量在 gather_dim 维度上拼接起来，并返回连续的张量
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """
    All-to-all 通信的 Autograd 函数。

    参数:
        input_ (torch.Tensor): 输入矩阵。
        process_group (dist.ProcessGroup): 通信进程组。
        scatter_dim (int): 分散操作的维度。
        gather_dim (int): 收集操作的维度。
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        """
        前向传播方法，执行 all-to-all 通信。

        参数:
            ctx: 上下文对象，用于在反向传播时传递信息。
            input_ (torch.Tensor): 输入张量。
            process_group (dist.ProcessGroup): 通信进程组。
            scatter_dim (int): 分散操作的维度。
            gather_dim (int): 收集操作的维度。

        返回:
            torch.Tensor: 收集后的张量。
        """
        # 将进程组、分散维度和收集维度存储在上下文中，以便反向传播时使用
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        # 获取进程组中的进程数量
        ctx.world_size = dist.get_world_size(process_group)

        # 执行 all-to-all 通信操作
        output = _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播方法，执行 all-to-all 通信以反向传播梯度。

        参数:
            ctx: 上下文对象，包含前向传播时存储的信息。
            grad_output (torch.Tensor): 输出梯度。

        返回:
            tuple: 反向传播的梯度，仅输入张量有梯度，其他参数无梯度。
        """
        # 执行 all-to-all 通信以反向传播梯度，交换 scatter_dim 和 gather_dim
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        # 返回反向传播的梯度，仅输入张量有梯度，其他参数无梯度
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    """
    执行 all-to-all 通信操作。

    参数:
        input_ (torch.Tensor): 输入张量。
        process_group (dist.ProcessGroup): 通信进程组。
        scatter_dim (int, optional): 分散操作的维度，默认为 2。
        gather_dim (int, optional): 收集操作的维度，默认为 1。

    返回:
        torch.Tensor: 收集后的张量。
    """
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)


def _gather(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    gather_dim: int,
):
    """
    执行 gather 通信操作，将输入张量从所有进程中收集到一个进程中。

    参数:
        input_ (torch.Tensor): 输入张量。
        world_size (int): 进程组中的进程数量。
        group (dist.ProcessGroup): 通信进程组。
        gather_dim (int): 收集操作的维度。

    返回:
        list[torch.Tensor]: 收集后的张量列表。
    """
    if gather_list is None:
        # 为每个进程创建一个空的输出张量列表，形状与 input_ 相同
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    # 执行 gather 操作，将 input_ 从所有进程中收集到 gather_list 中
    dist.gather(input_, gather_list, group=group, gather_dim=gather_dim)
    return gather_list


def _split(input_, pg: dist.ProcessGroup, dim=-1):
    """
    将输入张量沿指定维度分割，并返回当前进程的分割部分。

    参数:
        input_ (torch.Tensor): 输入张量，需要在指定维度上可以被 world_size 整除。
        pg (dist.ProcessGroup): 通信进程组。
        dim (int, optional): 分割操作的维度，默认为最后一个维度 (-1)。

    返回:
        torch.Tensor: 当前进程对应的分割后的张量。
    """
    # 如果进程组中只有一个进程，则无需分割，直接返回输入张量
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    # 获取指定维度的尺寸，并检查是否可以整除 world_size
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    # 沿指定维度将张量分割成 world_size 份
    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    # 获取当前进程对应的分割部分，并确保其是连续的
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_, pg: dist.ProcessGroup, dim=-1):
    """
    收集所有进程中的输入张量，并在指定维度上拼接成一个张量。

    参数:
        input_ (torch.Tensor): 输入张量。
        pg (dist.ProcessGroup): 通信进程组。
        dim (int, optional): 拼接操作的维度，默认为最后一个维度 (-1)。

    返回:
        torch.Tensor: 收集并拼接后的张量。
    """
    # 如果进程组中只有一个进程，则无需收集，直接返回输入张量
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    dist.get_rank(pg)

    if world_size == 1:
        return input_

    # 初始化一个空张量列表，用于存储来自所有进程的张量
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    # 检查输入张量是否在 CUDA 上，因为 all_gather 需要在 GPU 上操作
    assert input_.device.type == "cuda"
    # 执行 all_gather 操作，将所有进程中的输入张量收集到 tensor_list 中
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # 在指定维度上拼接收集到的张量，并返回连续的张量
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    在前向传播中收集输入张量并在模型并行区域拼接，在反向传播中分割梯度。

    参数:
        input_ (torch.Tensor): 输入矩阵。
        process_group (dist.ProcessGroup): 并行模式。
        dim (int): 操作的维度。
        grad_scale (str): 梯度缩放方式，可以是 "up" 或 "down"。
    """

    @staticmethod
    def symbolic(graph, input_):
        """
        符号函数，用于描述前向传播的操作。

        参数:
            graph: 计算图。
            input_ (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 收集后的张量。
        """
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        """
        前向传播方法，执行收集操作。

        参数:
            ctx: 上下文对象，用于在反向传播时传递信息。
            input_ (torch.Tensor): 输入张量。
            process_group (dist.ProcessGroup): 并行进程组。
            dim (int): 操作的维度。
            grad_scale (str): 梯度缩放方式。

        返回:
            torch.Tensor: 收集后的张量。
        """
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _gather(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播方法，执行分割操作。

        参数:
            ctx: 上下文对象，包含前向传播时存储的信息。
            grad_output (torch.Tensor): 输出梯度。

        返回:
            tuple: 反向传播的梯度，仅输入张量有梯度，其他参数无梯度。
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)

        return _split(grad_output, ctx.mode, ctx.dim), None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    在前向传播中分割输入张量并保留当前进程的对应部分，在反向传播中收集梯度。

    参数:
        input_ (torch.Tensor): 输入矩阵。
        process_group (dist.ProcessGroup): 并行模式。
        dim (int): 操作的维度。
    """

    @staticmethod
    def symbolic(graph, input_):
        """
        符号函数，用于描述前向传播的操作。

        参数:
            graph: 计算图。
            input_ (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 分割后的张量。
        """
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        """
        前向传播方法，执行分割操作。

        参数:
            ctx: 上下文对象，用于在反向传播时传递信息。
            input_ (torch.Tensor): 输入张量。
            process_group (dist.ProcessGroup): 并行进程组。
            dim (int): 操作的维度。
            grad_scale (str): 梯度缩放方式。

        返回:
            torch.Tensor: 分割后的张量。
        """
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _split(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播方法，执行收集操作。

        参数:
            ctx: 上下文对象，包含前向传播时存储的信息。
            grad_output (torch.Tensor): 输出梯度。

        返回:
            tuple: 反向传播的梯度，仅输入张量有梯度，其他参数无梯度。
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim), None, None, None


def split_forward_gather_backward(input_, process_group, dim, grad_scale=1.0):
    """
    在前向传播中分割输入张量并保留当前进程的对应部分，在反向传播中收集梯度。

    参数:
        input_ (torch.Tensor): 输入张量。
        process_group (dist.ProcessGroup): 并行进程组。
        dim (int): 操作的维度。
        grad_scale (float, optional): 梯度缩放因子，默认为 1.0。

    返回:
        torch.Tensor: 分割后的张量。
    """
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale)


def gather_forward_split_backward(input_, process_group, dim, grad_scale=None):
    """
    在前向传播中收集输入张量并在模型并行区域拼接，在反向传播中分割梯度。

    参数:
        input_ (torch.Tensor): 输入张量。
        process_group (dist.ProcessGroup): 并行进程组。
        dim (int): 操作的维度。
        grad_scale (float, optional): 梯度缩放因子，默认为 None。

    返回:
        torch.Tensor: 收集并拼接后的张量。
    """
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale)
