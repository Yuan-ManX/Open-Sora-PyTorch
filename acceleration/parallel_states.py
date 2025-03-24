import torch.distributed as dist


# 定义一个全局字典，用于存储不同类型的并行组
_GLOBAL_PARALLEL_GROUPS = dict()


def set_data_parallel_group(group: dist.ProcessGroup):
    """
    设置数据并行组。

    参数:
        group (dist.ProcessGroup): 要设置的数据并行进程组。
    """
    # 将数据并行组存储在全局字典中，键为 "data"
    _GLOBAL_PARALLEL_GROUPS["data"] = group


def get_data_parallel_group(get_mixed_dp_pg : bool = False):
    """
    获取数据并行组。

    参数:
        get_mixed_dp_pg (bool, optional): 是否获取混合数据并行组。如果为 True，则尝试获取名为 "mixed_dp_group" 的组。
                                           如果不存在，则返回默认的数据并行组。默认为 False。
    
    返回:
        dist.ProcessGroup: 数据并行进程组。如果未设置，则返回默认的 WORLD 组。
    """
    if get_mixed_dp_pg and "mixed_dp_group" in _GLOBAL_PARALLEL_GROUPS:
        # 如果需要获取混合数据并行组且该组存在于全局字典中，则返回混合数据并行组
        return _GLOBAL_PARALLEL_GROUPS["mixed_dp_group"]
    # 否则，返回默认的数据并行组（键为 "data"），如果不存在，则返回默认的 WORLD 组
    return _GLOBAL_PARALLEL_GROUPS.get("data", dist.group.WORLD)


def set_sequence_parallel_group(group: dist.ProcessGroup):
    """
    设置序列并行组。

    参数:
        group (dist.ProcessGroup): 要设置的序列并行进程组。
    """
    # 将序列并行组存储在全局字典中，键为 "sequence"
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group():
    """
    获取序列并行组。

    返回:
        dist.ProcessGroup | None: 序列并行进程组。如果未设置，则返回 None。
    """
    # 从全局字典中获取序列并行组，如果不存在，则返回 None
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)


def set_tensor_parallel_group(group: dist.ProcessGroup):
    """
    设置张量并行组。

    参数:
        group (dist.ProcessGroup): 要设置的张量并行进程组。
    """
    # 将张量并行组存储在全局字典中，键为 "tensor"
    _GLOBAL_PARALLEL_GROUPS["tensor"] = group


def get_tensor_parallel_group():
    """
    获取张量并行组。

    返回:
        dist.ProcessGroup | None: 张量并行进程组。如果未设置，则返回 None。
    """
    # 从全局字典中获取张量并行组，如果不存在，则返回 None
    return _GLOBAL_PARALLEL_GROUPS.get("tensor", None)

