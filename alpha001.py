from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union
from ops import rank, ts_argmax, stddev, signed_power

def alpha_001(close: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    实现 WorldQuant Alpha#1 因子。

    公式: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.0), 5)) - 0.5)

    该因子通过结合收益率的方向性、波动率（标准差）以及价格极值出现的时间点，
    在截面上对股票进行排序。它本质上是在捕捉价格动量与波动率挤压后的转折信号。

    参数
    ----------
    close : pd.DataFrame
        股票的收盘价数据。
        形状为 (n_days, n_stocks)，索引为日期，列为股票代码。
    returns : pd.DataFrame
        股票的日收益率数据（通常为 close.pct_change()）。
        形状必须与 close 一致。

    返回
    -------
    pd.DataFrame
        Alpha#1 因子值。
        取值范围在 [-0.5, 0.5] 之间，形状为 (n_days, n_stocks)。

    注意事项
    ----------
    1. 前 20 天的数据由于 stddev 计算会产生 NaN。
    2. Ts_ArgMax 窗口为 5，意味着需要至少 25 天的基础数据才能产生稳定的因子值。
    3. SignedPower 在幂次为 2 时，数学上等同于保留原始符号的平方。
    """

    # 1. 逻辑条件处理：(returns < 0) ? stddev(returns, 20) : close
    # stddev(returns, 20) 属于时间序列运算
    volatility_20d = stddev(returns, 20)

    # 使用 np.where 进行向量化条件选择
    # 如果收益率为负，取其 20 日波动率；否则取收盘价
    inner_value = np.where(returns < 0, volatility_20d, close)

    # 将 numpy 结果包装回 DataFrame 以保持索引对齐
    inner_df = pd.DataFrame(inner_value, index=close.index, columns=close.columns)

    # 2. SignedPower(..., 2.0)
    # 保持符号的幂运算，增强特征的非线性表达
    power_val = signed_power(inner_df, 2.0)

    # 3. Ts_ArgMax(..., 5)
    # 寻找过去 5 个交易日内，power_val 最大值出现的相对位置（0-4）
    # 这是一个时间序列算子，反映了近期最强信号的“新鲜度”
    argmax_val = ts_argmax(power_val, 5)

    # 4. rank(...) - 0.5
    # 截面排名（Cross-sectional Rank）：将所有股票在同一天的值进行百分比排序
    # 减去 0.5 实现中心化（Mean Neutralization），使其符合多空对冲策略的输入要求
    alpha_values = rank(argmax_val) - 0.5

    return alpha_values
