from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union
# 假设 ops 模块中包含这些基础运算
from ops import rank, delta, correlation


def alpha_002(open: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    实现 WorldQuant Alpha#2 因子。

    公式: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))

    该因子衡量的是交易量变化的动量与日内收益率在截面排名上的相关性。
    通常用于捕捉价量背离或同步的信号，负号表示对这种相关性进行反向操作。

    参数
    ----------
    open : pd.DataFrame
        股票的开盘价数据。
    close : pd.DataFrame
        股票的收盘价数据。
    volume : pd.DataFrame
        股票的成交量数据。

    返回
    -------
    pd.DataFrame
        Alpha#2 因子值。形状为 (n_days, n_stocks)。

    注意事项
    ----------
    1. delta(log(volume), 2) 至少需要 3 天数据（包含 log 转换）。
    2. correlation(..., 6) 窗口为 6，加上 delta 的延迟，通常需要前 8 天数据产生有效值。
    3. 该因子反映了量价关系的短期演变。
    """

    # 1. 计算成交量的变化率排名：rank(delta(log(volume), 2))
    # 首先取对数平滑极端值，然后计算 2 日差分
    log_vol = np.log(volume)
    vol_delta = delta(log_vol, 2)
    # 对差分结果进行截面排名
    rank_vol_delta = rank(vol_delta)

    # 2. 计算日内收益率排名：rank(((close - open) / open))
    # 计算当日从开盘到收盘的收益率
    intra_day_returns = (close - open) / open
    # 对日内收益率进行截面排名
    rank_intra_ret = rank(intra_day_returns)

    # 3. 计算两者的 6 日滚动相关性：correlation(..., 6)
    # 这是一个时间序列运算
    corr_6d = correlation(rank_vol_delta, rank_intra_ret, 6)

    # 4. 最后乘以 -1
    alpha_values = -1 * corr_6d

    return alpha_values
