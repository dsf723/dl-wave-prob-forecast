"""CDIP 浮标海浪谱 → 实测能量周期 Te 等参数（功率矩阵 Hs × Te 要用的真实周期，ERA5 / GEFS 的 Te 只能当预报或近似）。

输入：raw/observations/cdip/<站号>_historic.nc（NCEI 归档的 CDIP 全部投放合并文件：121p1 关岛 Ipan、198p1 夏威夷
     Kaneohe WETS、196p1 关岛 Ritidian；10-22 下载）
计算（只用 waveFlagPrimary == 1 的谱，约每 30 分钟一条）：
  m_n = Σ S(f)·f^n·Δf（S = waveEnergyDensity，Δf = waveBandwidth）
  hs_spec = 4√m0；te = m₋₁/m0（能量周期，波能流 P = 0.49·Hs²·Te 用的就是它）；
  tm01 = m0/m1；tm02 = √(m0/m2)；tp = waveTp；dp = waveDp（峰向）；
  mdir = 能量加权平均来向（由 a1、b1：atan2(Σb1·S·Δf, Σa1·S·Δf)，CDIP 约定为"来自"方向，°T）
  再按整点 ±30 分钟取平均（方向用圆平均）。
输出：processed/buoy_spectral/<站点ID>_<站号>.csv（valid_time, hs_spec, te, tm01, tm02, tp, dp, mdir, n）
用法：python cdip_energy_period.py [G:121p1 H:198p1 G2:196p1]
"""
import os
import sys

import netCDF4
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'raw', 'observations', 'cdip')
OUT = os.path.join(ROOT, 'processed', 'buoy_spectral')


def convert(site, station):
    path = os.path.join(RAW, f'{station}_historic.nc')
    with netCDF4.Dataset(path) as d:
        ok = d['waveFlagPrimary'][:] == 1
        t = pd.to_datetime(np.asarray(d['waveTime'][:])[ok], unit='s', utc=True)
        f = np.asarray(d['waveFrequency'][:], float)
        df = np.asarray(d['waveBandwidth'][:], float)
        s = np.ma.filled(d['waveEnergyDensity'][:][ok], np.nan).astype(float)
        a1 = np.ma.filled(d['waveA1Value'][:][ok], np.nan).astype(float)
        b1 = np.ma.filled(d['waveB1Value'][:][ok], np.nan).astype(float)
        tp, dp = (np.ma.filled(d[v][:][ok], np.nan).astype(float) for v in ('waveTp', 'waveDp'))
    e = s * df
    m0, m1, m2, mm1 = e.sum(1), (e * f).sum(1), (e * f ** 2).sum(1), (e / f).sum(1)
    mdir = (np.degrees(np.arctan2(np.nansum(b1 * e, 1), np.nansum(a1 * e, 1))) + 360) % 360
    x = pd.DataFrame({'hs_spec': 4 * np.sqrt(m0), 'te': mm1 / m0, 'tm01': m0 / m1, 'tm02': np.sqrt(m0 / m2),
                      'tp': tp, 'dp': dp, 'sin': np.sin(np.radians(mdir)), 'cos': np.cos(np.radians(mdir))}, index=t)
    x = x[np.isfinite(x.te) & (x.hs_spec > 0)]
    hour = x.index.round('h')
    h = x.groupby(hour).mean()
    h['mdir'] = (np.degrees(np.arctan2(h.pop('sin'), h.pop('cos'))) + 360) % 360
    h['n'] = x.groupby(hour).size()
    h.index.name = 'valid_time'
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f'{site}_{station}.csv')
    h.round(3).reset_index().assign(valid_time=lambda z: z.valid_time.dt.strftime('%Y-%m-%dT%H:%M:%SZ')).to_csv(out, index=False)
    print(f'[完成] {site} CDIP {station}：{h.index.min():%Y-%m-%d} – {h.index.max():%Y-%m-%d}，{len(h)} 个小时；'
          f'Te 均值 {h.te.mean():.2f} s（Te/Tp {np.nanmedian(h.te / h.tp):.2f}，Te/Tm02 {np.nanmedian(h.te / h.tm02):.2f}），'
          f'Hs 均值 {h.hs_spec.mean():.2f} m → {out}', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for item in (sys.argv[1:] or ['G:121p1', 'H:198p1', 'G2:196p1']):
        site, station = item.split(':')
        if os.path.exists(os.path.join(RAW, f'{station}_historic.nc')):
            convert(site, station)
        else:
            print(f'[跳过] {station}_historic.nc 还没下载')


if __name__ == '__main__':
    main()
