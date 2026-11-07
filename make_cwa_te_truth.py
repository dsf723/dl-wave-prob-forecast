"""新竹浮标的能量周期 Te 真值（11-04 事先登记的第 3 条）→ processed/buoy_spectral_cwa/CW46757B_46757B.csv（valid_time, te）。

气象署历史资料只有尖峰周期 Tp 和平均周期 Tmean（定义未写明）。事先定的判定：训练年 2000–2015 的 Tmean / ERA5 Te 中位数在 0.78–0.90 之间，
视为 Tmean ≈ Tm02，用一个常数换成 Te。新竹 0.863（通过）；马祖 0.926（未通过，不做周期真值，只作参考）。
换算：Te = Tmean ÷ k，k = 训练年（2000–2015）Tmean / ERA5 Te 的中位数，即以 ERA5 Te 为尺度。代价：Te 的平均偏差按构造为 0，
      周期检验只看起伏和时间变化，不检验绝对偏差。k 只用训练年拟合，验证、测试年不参与。
用法：python make_cwa_te_truth.py
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'processed', 'buoy_spectral_cwa')
SID, ST, ERA5 = 'CW46757B', '46757B', 'CW46757B'
TRAIN = (2000, 2015)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    b = pd.read_csv(os.path.join(ROOT, 'processed', 'buoy', f'{SID}_{ST}.csv'), usecols=['valid_time', 'tmean_cwa'])
    b.index = pd.to_datetime(b.valid_time.str.replace('Z', ''))
    e = pd.read_csv(os.path.join(ROOT, 'processed', f'site_{ERA5}', 'observations_hourly.csv'), usecols=['valid_time', 'te'])
    e.index = pd.to_datetime(e.valid_time.str.replace('Z', ''))
    x = pd.DataFrame({'tm': b.tmean_cwa, 'te': e.te}).dropna()
    x = x[(x.index.year >= TRAIN[0]) & (x.index.year <= TRAIN[1])]
    k = float((x.tm / x.te).median())
    if not .78 <= k <= .90:
        raise SystemExit(f'k = {k:.3f} 不在事先定的 0.78–0.90 内，不做 Te 真值')
    o = pd.DataFrame({'valid_time': b.valid_time, 'te': (b.tmean_cwa / k).round(3)}).dropna()
    os.makedirs(OUT, exist_ok=True)
    o.to_csv(os.path.join(OUT, f'{SID}_{ST}.csv'), index=False)
    json.dump(dict(site=SID, k_tmean_over_era5_te=k, n_fit=len(x), fit_years=list(TRAIN), rule='Te = Tmean / k; k 在 0.78–0.90 才做',
                   note='气象署平均周期定义未写明；以 ERA5 Te 为尺度，平均偏差按构造为 0'),
              open(os.path.join(OUT, 'te_conversion.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{SID}：k = {k:.3f}（{len(x):,} 个训练年配对），Te 真值 {len(o):,} 行 → {OUT}')


if __name__ == '__main__':
    main()
