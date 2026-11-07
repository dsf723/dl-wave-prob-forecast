"""30 天产品的后半段（第 16–30 天用气候态）：比较几种气候态，只在验证集上评价。

候选（都输出 99 个分位数，用与模型相同的 CRPS 评分；真值 = 浮标 Hs）：
  M0 浮标·逐月      该站训练年份浮标实测、按日历月（= 目前 export_quantile_forecasts.py --days 30 的做法）
  M1 浮标·日窗      训练年份浮标实测、预见时刻所在日期 ±15 天
  M2 ERA5订正·日窗  ERA5 单点长序列（1940 起），先用训练年份的"ERA5 ↔ 浮标"分位数映射订正到浮标尺度，
                    再取日期 ±15 天；年份用全部 ERA5 年份，但去掉验证、测试年份
  M3 M2 + ENSO      只取与起报时 ENSO 状态相同（ONI ≥ 0.5 厄尔尼诺 / ≤ −0.5 拉尼娜 / 其余中性）的年份季节
  M5 M1 与 M2 平均  两者分位数逐个平均（浮标近年 + ERA5 长期）
  M4 M2 + MJO 类比  在 1974 年以后的 ERA5 订正序列里找"历史起报日"：日期 ±15 天、MJO 状态相同（振幅 < 1 记为弱，
                    否则按 1–8 位相），取它们之后同一预见时效的值（类比集合）；类比日少于 20 个时退回 M2
起报时的 ENSO / MJO 只用起报前已发布的数值（ONI 取上一个月结束的三个月季节；RMM 取决策前一天）。
评价起报：验证年份里的每个周三 00 UTC，决策时刻 = 起报 + 6 h，评价第 16–30 天（决策后 +363 h 至 +720 h）、
  且这些时刻全落在验证年份内的起报。显著性：按起报配对的自助法（2000 次）。
另给出验证期的逐月平均 Hs 与 MVI（(最大月 − 最小月) / 平均）：看各气候态能否复现季节变化。
输出：results/climatology_days16_30/scores.csv、monthly.csv、report.md
用法：python climatology_days16_30.py [G K H]
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from era5_to_observations import SITES, load_site_file   # noqa: E402
from wavecast.metrics import score, TAU                  # noqa: E402

CFG = os.path.join(ROOT, 'model_configs', 'GKH_h15_transformer_pretrain.json')
OUT = os.path.join(ROOT, 'results', 'climatology_days16_30')
STEP, FIRST, LAST = 3, 121, 240                          # 第 16 天 ~ 第 30 天（决策后的 3 小时步）
WINDOW, MIN_ANALOG = 15, 20
NAMES = {'M0': '浮标·逐月（现行）', 'M1': '浮标·日窗', 'M2': 'ERA5订正·日窗', 'M3': 'ERA5订正+ENSO', 'M4': 'ERA5订正+MJO类比',
         'M5': '浮标日窗与ERA5订正平均'}


def doy_dist(a, b):
    d = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    return np.minimum(d, 366 - d)


def era5_long(site):
    lat, lon = site['point']
    files = sorted(f for f in os.listdir(os.path.join(ROOT, 'era5_data')) if f.startswith(f"ts_{site['name']}_"))
    parts, used = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            parts += load_site_file(os.path.join(ROOT, 'era5_data', f), lat, lon, tmp, used)
    hs = pd.concat(parts)['hs'].groupby(level=0).first().sort_index()
    hs.index = hs.index.tz_localize('UTC') if hs.index.tz is None else hs.index
    return hs[hs.index.hour % 3 == 0].dropna()


def quantile_map(era5, buoy):
    p = np.linspace(.001, .999, 199)
    qe, qb = np.quantile(era5, p), np.quantile(buoy, p)
    def f(x):
        y = np.interp(x, qe, qb)
        hi, lo = x > qe[-1], x < qe[0]
        y[hi] = qb[-1] * x[hi] / qe[-1]                    # 尾部按比例外推
        y[lo] = qb[0] * x[lo] / max(qe[0], 1e-6)
        return y
    return f


def oni_table():
    t = pd.read_csv(os.path.join(ROOT, 'raw', 'indices', 'oni.ascii.txt'), sep=r'\s+')
    centre = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}
    t['month'] = t.SEAS.map(centre)
    t['date'] = pd.to_datetime(dict(year=t.YR, month=t.month, day=15), utc=True)
    s = t.set_index('date').ANOM
    return s


def enso_class(oni, when):
    """ENSO state known at `when`: season centred one month before the previous month (ends last month)."""
    v = oni.reindex(pd.DatetimeIndex(when).floor('D') - pd.Timedelta(days=45), method='ffill').to_numpy()
    return np.where(v >= .5, 1, np.where(v <= -.5, -1, 0))


def rmm_table():
    t = pd.read_csv(os.path.join(ROOT, 'raw', 'indices', 'rmm.74toRealtime.txt'), sep=r'\s+', skiprows=2, header=None,
                    usecols=[0, 1, 2, 5, 6], names=['y', 'm', 'd', 'phase', 'amp'])
    t = t[(t.amp < 100) & (t.phase.between(1, 8))]
    idx = pd.to_datetime(dict(year=t.y, month=t.m, day=t.d), utc=True)
    return pd.Series(np.where(t.amp.to_numpy() < 1, 0, t.phase.to_numpy()), index=idx)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cfg = json.load(open(CFG, encoding='utf-8'))
    splits = {s['site_id']: s for s in cfg['sites']}
    oni, rmm = oni_table(), rmm_table()
    rows, monthly = [], []
    for sid in (sys.argv[1:] or ['G', 'K', 'H']):
        sp, site = splits[sid], SITES[sid] if isinstance(SITES, dict) else next(s for s in SITES if s['id'] == sid)
        train, banned = set(sp['train_years']), set(sp['val_years']) | set(sp['test_years'])
        obs = pd.read_csv(os.path.join(ROOT, 'processed', f'site_{sid}_reforecast', 'observations.csv'),
                          usecols=['valid_time', 'hs'], parse_dates=['valid_time']).set_index('valid_time').hs.dropna()
        e5 = era5_long(site)
        ov = pd.concat([e5, obs], axis=1, keys=['e', 'b']).dropna()
        ov = ov[ov.index.year.isin(train)]
        qm = quantile_map(ov.e.to_numpy(), ov.b.to_numpy())
        e5c = pd.Series(qm(e5.to_numpy()), index=e5.index)
        e5c = e5c[~e5c.index.year.isin(banned)]
        btr = obs[obs.index.year.isin(train)]
        print(f'{sid}：浮标训练年 {min(train)}–{max(train)}（{len(btr)} 个 3 小时值）；ERA5 订正序列 {e5c.index.year.min()}–'
              f'{e5c.index.year.max()}，{e5c.index.year.nunique()} 年（已去掉验证/测试年）；映射样本 {len(ov)}', flush=True)

        # evaluation issues: Wednesdays 00 UTC in validation years with all of days 16–30 inside validation years
        days = pd.date_range(f"{min(sp['val_years'])}-01-01", f"{max(sp['val_years'])}-12-31", freq='D', tz='UTC')
        issues = [d for d in days[days.dayofweek == 2]
                  if (d + pd.Timedelta(hours=6 + LAST * STEP)).year in sp['val_years']]
        lead_h = np.arange(FIRST, LAST + 1) * STEP

        def table(series, doys):
            v, dy = series.to_numpy(), series.index.dayofyear.to_numpy()
            return {d: np.quantile(v[np.minimum(abs(dy - d), 366 - abs(dy - d)) <= WINDOW], TAU) for d in doys}

        month_q = {m: np.quantile(btr[btr.index.month == m], TAU) for m in range(1, 13)}
        b_doy = table(btr, range(1, 367))
        e_doy = table(e5c, range(1, 367))
        e_years = pd.Series(e5c.index.year, index=e5c.index)
        cls_hist = pd.Series(enso_class(oni, e5c.index), index=e5c.index)
        mjo_hist = rmm.reindex(pd.date_range('1974-06-01', '2026-10-20', freq='D', tz='UTC'))
        analog_days = mjo_hist.dropna()
        analog_days = analog_days[~analog_days.index.year.isin(banned)]
        e5c_day06 = e5c  # 3-hourly series used with exact lead offsets
        enso_cache = {}
        for it in issues:
            dec = it + pd.Timedelta(hours=6)
            t = dec + pd.to_timedelta(lead_h, unit='h')
            y = obs.reindex(t).to_numpy()
            ok = np.isfinite(y)
            if ok.sum() < 20:
                continue
            doy = t.dayofyear.to_numpy()
            q = {'M0': np.stack([month_q[m] for m in t.month]), 'M1': np.stack([b_doy[d] for d in doy]),
                 'M2': np.stack([e_doy[d] for d in doy])}
            c = int(enso_class(oni, [dec])[0])
            key = (c,)
            if key not in enso_cache:
                enso_cache[key] = table(e5c[cls_hist.to_numpy() == c], range(1, 367))
            q['M3'] = np.stack([enso_cache[key][d] for d in doy])
            phase = rmm.reindex([dec.floor('D') - pd.Timedelta(days=1)]).iloc[0]
            q4 = q['M2'].copy()
            if pd.notna(phase):
                cand = analog_days[(analog_days == phase) &
                                   (doy_dist(analog_days.index.dayofyear, [dec.dayofyear])[:, 0] <= WINDOW)].index
                anchors = cand + pd.Timedelta(hours=6)
                if len(anchors) >= MIN_ANALOG:
                    for j, L in enumerate(lead_h):
                        vals = e5c.reindex(anchors + pd.Timedelta(hours=int(L))).dropna().to_numpy()
                        if len(vals) >= MIN_ANALOG:
                            q4[j] = np.quantile(vals, TAU)
            q['M4'] = q4
            q['M5'] = (q['M1'] + q['M2']) / 2
            for k, qq in q.items():
                cr = score(qq[ok], y[ok])
                rows.append(dict(site=sid, issue=it, method=k, crps=cr.mean(), n=ok.sum()))
                monthly.append(pd.DataFrame({'site': sid, 'method': k, 'month': t.month[ok], 'pred_mean': qq[ok].mean(1),
                                             'obs': y[ok]}))
    sc = pd.DataFrame(rows)
    mo = pd.concat(monthly)
    os.makedirs(OUT, exist_ok=True)
    sc.to_csv(os.path.join(OUT, 'scores.csv'), index=False)
    mon = mo.groupby(['site', 'method', 'month'])[['pred_mean', 'obs']].mean().reset_index()
    mon.to_csv(os.path.join(OUT, 'monthly.csv'), index=False, encoding='utf-8-sig')

    rng = np.random.default_rng(0)
    lines = ['# 第 16–30 天气候态比较（验证集）', '',
             '| 站点 | 方法 | 起报数 | CRPS | 相对 M0 | 95% 区间 | 显著 | 预报 MVI | 实测 MVI |', '|---|---|---:|---:|---:|---|---|---:|---:|']
    for sid in sc.site.unique():
        w = sc[sc.site == sid].pivot(index='issue', columns='method', values='crps')
        for k in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
            rel = w[k].mean() / w['M0'].mean() - 1
            idx = rng.integers(0, len(w), (2000, len(w)))
            boot = w[k].to_numpy()[idx].mean(1) / w['M0'].to_numpy()[idx].mean(1) - 1
            lo, hi = np.percentile(boot, [2.5, 97.5])
            m = mon[(mon.site == sid) & (mon.method == k)]
            mvi_p = (m.pred_mean.max() - m.pred_mean.min()) / m.pred_mean.mean()
            mvi_o = (m.obs.max() - m.obs.min()) / m.obs.mean()
            sig = '—' if k == 'M0' else ('是' if hi < 0 or lo > 0 else '否')
            lines.append(f'| {sid} | {k} {NAMES[k]} | {len(w)} | {w[k].mean():.4f} | {rel:+.1%} | [{lo:+.1%}, {hi:+.1%}] | {sig} | '
                         f'{mvi_p:.2f} | {mvi_o:.2f} |')
    lines += ['', 'CRPS 越小越好；相对 M0 为负 = 比现行做法好。MVI 按验证期逐月平均 Hs 计算（Hs 的 MVI，不是波能流）。',
              '验证期只有 2–3 年，逐月实测本身噪声大，MVI 只作参考。']
    open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
