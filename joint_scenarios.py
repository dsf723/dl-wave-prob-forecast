"""波高–周期成对情景（给波能流和最后的功率矩阵用）。

波高 Hs 和能量周期 Te 分别订正（两个模型、各自的 99 个分位数），但功率 / 波能流取决于两者的组合：
大浪往往周期也长（正相关），分开取 P50 相乘会低估均值。这里用高斯 copula 把两条边缘分布配成对：
  - 相关程度：训练年份里浮标实测 Hs 与实测 Te（CDIP 谱）同一时刻的 Spearman 秩相关 r_s，按站点 × 月份估计，
    换算成高斯相关 ρ = 2·sin(π·r_s / 6)；
  - 抽样：z1、z2 为标准正态（固定随机种子，所有时刻共用同一批随机数），z_te = ρ·z1 + √(1−ρ²)·z2，
    u = Φ(z)，Hs = Q_Hs(u1)、Te = Q_Te(u2)（分位数函数在 99 个概率点之间线性插值，两端取端点值）。
每个时刻只配同一时刻的 Hs–Te（不处理预见期之间的时间相关）；要做"一段时间的发电量"时，再按集合成员秩次
或历史轨迹（Schaake 洗牌）补时间结构。
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

TAU = (np.arange(99) + .5) / 99


def copula_rho(hs, te, min_pairs=200):
    """{month: ρ} from paired observations (pandas Series on the same UTC index); NaN pairs dropped."""
    d = pd.DataFrame({'hs': hs, 'te': te}).dropna()
    out = {}
    for m in range(1, 13):
        g = d[d.index.month == m]
        if len(g) < min_pairs:
            g = d                                                   # 样本太少：用全年
        rs = spearmanr(g.hs, g.te).statistic
        out[m] = float(2 * np.sin(np.pi * rs / 6))
    return out


def _inverse(qrows, u):
    """Quantile functions (rows of 99 values at TAU) evaluated at probabilities u (n,) → (rows, n)."""
    pos = np.clip(u * 99 - .5, 0, 98)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, 98)
    w = pos - lo
    return qrows[:, lo] * (1 - w) + qrows[:, hi] * w


def joint_samples(hs_q, te_q, rho, n=200, seed=0):
    """hs_q, te_q: (rows, 99) quantiles; rho: (rows,) Gaussian correlation. Returns Hs, Te samples (rows, n)."""
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    hs = _inverse(np.asarray(hs_q), norm.cdf(z1))
    rho = np.asarray(rho)[:, None]
    zt = rho * z1[None, :] + np.sqrt(1 - rho ** 2) * z2[None, :]
    pos = np.clip(norm.cdf(zt) * 99 - .5, 0, 98)                   # 每行 ρ 不同：按行取各自的概率位置
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, 98)
    w = pos - lo
    tq = np.asarray(te_q)
    te = np.take_along_axis(tq, lo, 1) * (1 - w) + np.take_along_axis(tq, hi, 1) * w
    return hs, te
