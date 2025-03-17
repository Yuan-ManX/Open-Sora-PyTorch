from copy import deepcopy
import torch.nn as nn
from mmengine.registry import Registry


# 定义一个装饰器，用于注册模块的注册表类
def build_module(module: dict | nn.Module, builder: Registry, **kwargs) -> nn.Module | None:
    """
    根据配置字典构建模块，或者直接返回模块本身。

    参数:
        module (dict | nn.Module): 要构建的模块。可以是配置字典或nn.Module实例。
        builder (Registry): 用于构建模块的注册表。
        **kwargs: 传递给构建函数的附加关键字参数。

    返回:
        (None | nn.Module): 创建的模型。如果输入为None，则返回None。
    """
    if module is None:
        return None
    if isinstance(module, dict):
        # 如果模块是字典，则进行深度复制以避免修改原始配置
        cfg = deepcopy(module)
        # 将额外的关键字参数添加到配置中
        for k, v in kwargs.items():
            cfg[k] = v
        # 使用注册表构建模块
        return builder.build(cfg)
    elif isinstance(module, nn.Module):
        # 如果模块已经是nn.Module实例，则直接返回该实例
        return module
    elif module is None:
        return None
    else:
        raise TypeError(f"Only support dict and nn.Module, but got {type(module)}.")


# 创建两个注册表实例，分别用于模型和数据集
MODELS = Registry(
    "model",
    locations=["opensora.models"],
)

DATASETS = Registry(
    "dataset",
    locations=["opensora.datasets"],
)
