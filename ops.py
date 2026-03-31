from __future__ import annotations  # 必须在第一行，用于支持延迟类型注解
import numpy as np
import pandas as pd

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """横截面排名：将值映射到 [0, 1] 区间"""
    return df.rank(axis=1, pct=True)

def ts_argmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列算子：返回过去 window 天内最大值出现的索引位置（从 0 开始）"""
    # 这里的 np.argmax 需要 numpy
    return df.rolling(window).apply(np.argmax, raw=True)

def stddev(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列算子：滚动标准差"""
    return df.rolling(window).std()

def signed_power(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """保持符号的幂运算"""
    # 这里的 np.sign 和 np.abs 需要 numpy
    return np.sign(df) * (np.abs(df) ** p)

def delta(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """时间序列算子：计算当前值与 period 天前的值之差"""
    return df.diff(period)

def correlation(df1: pd.DataFrame, df2: pd.DataFrame, window: int) -> pd.DataFrame:
    """时间序列算子：滚动皮尔逊相关系数"""
    return df1.rolling(window).corr(df2)
