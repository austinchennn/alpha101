import numpy as np
import pandas as pd

def rank(df):
    """横截面排名：将值映射到 [0, 1] 区间"""
    return df.rank(axis=1, pct=True)

def ts_argmax(df, window):
    """时间序列算子：返回过去 window 天内最大值出现的索引位置（从 0 开始）"""
    return df.rolling(window).apply(np.argmax, raw=True)

def stddev(df, window):
    """时间序列算子：滚动标准差"""
    return df.rolling(window).std()

def signed_power(df, p):
    """保持符号的幂运算"""
    return np.sign(df) * (np.abs(df) ** p)
