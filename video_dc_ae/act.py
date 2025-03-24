from functools import partial
from typing import Optional
import torch.nn as nn

from vo_ops import build_kwargs_from_config


# 定义一个列表，包含所有公开的接口名称
__all__ = ["build_act"]


# 注册激活函数的字典，键为激活函数的名称，值为对应的激活函数类
# 例如，"relu" 对应 nn.ReLU 类，"gelu" 对应 nn.GELU 类，并指定 approximate 参数为 "tanh"
REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    """
    根据激活函数名称和参数构建激活函数模块。

    参数:
        name (str): 激活函数的名称，必须在 REGISTERED_ACT_DICT 中注册。
        **kwargs: 其他传递给激活函数类的关键字参数。

    返回:
        Optional[nn.Module]: 如果激活函数名称有效，则返回对应的激活函数模块；否则，返回 None。
    """
    if name in REGISTERED_ACT_DICT:
        # 从注册表中获取激活函数类
        act_cls = REGISTERED_ACT_DICT[name]
        # 构建传递给激活函数类的关键字参数
        args = build_kwargs_from_config(kwargs, act_cls)
        # 实例化激活函数类，并返回激活函数模块
        return act_cls(**args)
    else:
        return None
