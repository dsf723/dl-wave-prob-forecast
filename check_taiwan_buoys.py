"""台湾海域 14 个浮标（Copernicus INSITU_GLO_WAV_DISCRETE_MY_013_045，逐时）的数据质量，以及与 GEFS 回算的配对。

输入：raw/insitu_taiwan_cmems/GL_TS_MO_*.nc（VGHS + VGHS_QC，QC = 1 算好数据）；
      processed/gefs_reforecast/<站号>/<年>.csv.gz（本机只有 46694A、46757B、C6W08，周三，11 成员）。
内容：每站好数据比例、最长缺测、逐年覆盖率（按 3 小时格点）、波高分辨率、≥ 2.5 m 个数；
      有回算的站：集合平均与浮标配对（时效 3–24 h），偏差、RMSE、相关；并数与回算重叠的周三起报里、前 15 天真值完整度 ≥ 80% 的次数。
对照行 K、G、H：同一算法、2012–2017 年。只描述数据，不训练。输出：results/taiwan_buoys/quality.md
用法：python check_taiwan_buoys.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from prepare_reforecast_dataset import read_gz_tolerant

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'results', 'taiwan_buoys', 'quality.md')
PINGTAN = (25.5, 120.0)
ZHOUSHAN = (30.5, 123.0)


def dist(a, b):
    p = np.radians
    x = np.sin(p(b[0] - a[0]) / 2) ** 2 + np.cos(p(a[0])) * np.cos(p(b[0])) * np.sin(p(b[1] - a[1]) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(x))


def read_buoy(path):
    x = xr.open_dataset(path)
    t = pd.to_datetime(x['TIME'].values)
    hs = x['VGHS'].values[:, 0].astype(float)
    qc = x['VGHS_QC'].values[:, 0].astype(float)
    meta = dict(name=x.attrs.get('platform_name', ''), wmo=x.attrs.get('wmo_platform_code', ''), source=x.attrs.get('source', ''),
                lat=float(x['LATITUDE'].values), lon=float(x['LONGITUDE'].values), depth=float(np.nanmean(x['DEPH'].values)))
    x.close()
    return pd.DataFrame(dict(hs=hs, qc=qc), index=t), meta


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    rows, cover, pair = [], {}, []
    for path in sorted(glob.glob(os.path.join(ROOT, 'raw', 'insitu_taiwan_cmems', 'GL_TS_MO_*.nc'))):
        sid = os.path.basename(path)[9:-3]
        d, m = read_buoy(path)
        good = d[(d.qc == 1) & d.hs.notna()].hs
        good = good[~good.index.duplicated()].sort_index()
        grid = pd.date_range(d.index.min().ceil('3h'), d.index.max().floor('3h'), freq='3h')
        s3 = good.reindex(grid)
        gaps = good.index.to_series().diff().dt.total_seconds() / 86400
        vals = np.unique(np.round(good.to_numpy(), 3))
        rows.append(dict(id=sid, name=m['name'], lat=m['lat'], lon=m['lon'], start=f'{d.index.min():%Y-%m}', end=f'{d.index.max():%Y-%m}',
                         n=len(good), qc_ok=len(good) / max(len(d), 1), cover=s3.notna().mean(), gap=gaps.max(), res=np.median(np.diff(vals)) if len(vals) > 1 else np.nan,
                         mean=good.mean(), p99=good.quantile(.99), max=good.max(), max_t=good.idxmax(), jump=int((good.diff().abs()[gaps <= 1 / 24 + 1e-6] > 2).sum()), big=int((s3 >= 2.5).sum()), d_p=dist(PINGTAN, (m['lat'], m['lon'])), d_z=dist(ZHOUSHAN, (m['lat'], m['lon']))))
        cover[sid] = s3.notna().groupby(s3.index.year).mean()
        g = sorted(glob.glob(os.path.join(ROOT, 'processed', 'gefs_reforecast', sid, '*.csv.gz')))
        if g:
            years = sorted(set(good.index.year))
            f = pd.concat([read_gz_tolerant(p)[0] for p in g if int(os.path.basename(p)[:4]) in years])
            f['issue_time'] = pd.to_datetime(f.issue_time).dt.tz_localize(None)
            f['valid_time'] = pd.to_datetime(f.valid_time).dt.tz_localize(None)
            f = f.drop_duplicates(['issue_time', 'member', 'valid_time'])
            f = f[f.issue_time.between(good.index.min(), good.index.max())]
            f['lead'] = (f.valid_time - f.issue_time).dt.total_seconds() / 3600
            e = f[f.lead.between(3, 24)].groupby(['issue_time', 'valid_time']).hs.mean().reset_index()
            e['obs'] = good.reindex(e.valid_time).to_numpy()
            e = e.dropna()
            issues = f.issue_time.drop_duplicates()
            okn = 0
            for it in issues:
                need = pd.date_range(it + pd.Timedelta('3h'), it + pd.Timedelta('15D'), freq='3h')
                okn += good.reindex(need).notna().mean() >= .8
            pair.append(dict(id=sid, name=m['name'], n=len(e), obs=e.obs.mean(), gefs=e.hs.mean(), bias=(e.hs - e.obs).mean(), rmse=np.sqrt(((e.hs - e.obs) ** 2).mean()),
                             r=np.corrcoef(e.hs, e.obs)[0, 1], issues=len(issues), ok=int(okn)))
    r = pd.DataFrame(rows).sort_values('d_p')
    L = ['# 台湾海域 14 个浮标的数据质量', '', 'Copernicus 逐时产品，只用 QC = 1。覆盖率按 3 小时格点算。平潭、舟山距离指到站点 C（25.5°N、120.0°E）和 A（30.5°N、123.0°E）。', '',
         '| 站号 | 名称 | 纬度 | 经度 | 距平潭 km | 距舟山 km | 起止 | 好数据时刻 | QC 通过 | 覆盖率 | 最长缺测（天）| 分辨率 m | 平均 Hs | P99 | 最大 | 最大值时刻 | 1 小时跳变 > 2 m | ≥ 2.5 m 时刻 |',
         '|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    L += [f'| {x.id} | {x["name"]} | {x.lat:.2f} | {x.lon:.2f} | {x.d_p:.0f} | {x.d_z:.0f} | {x.start}–{x.end} | {x.n:,} | {x.qc_ok:.0%} | {x.cover:.0%} | {x.gap:.0f} | {x.res:.2f} | {x["mean"]:.2f} | {x.p99:.2f} | {x["max"]:.2f} | {x.max_t:%Y-%m-%d} | {x.jump} | {x.big:,} |'
          for _, x in r.iterrows()]
    c = pd.DataFrame(cover).T.reindex(r.id)
    L += ['', '## 逐年覆盖率（3 小时格点）', '', '| 站号 | ' + ' | '.join(str(y) for y in c.columns) + ' |', '|---|' + '---:|' * len(c.columns)]
    L += [f'| {i} | ' + ' | '.join('' if pd.isna(v) else f'{v:.0%}' for v in row) + ' |' for i, row in zip(c.index, c.to_numpy())]
    L += ['', '## 与本机 GEFS 回算配对（集合平均，时效 3–24 h）', '', '| 站号 | 名称 | 配对数 | 浮标平均 | GEFS 平均 | 偏差 | RMSE | 相关 | 重叠的周三起报 | 其中前 15 天真值 ≥ 80% |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sid, label in (('K', 'K 马罗岛（对照）'), ('G', 'G 关岛（对照）'), ('H', 'H 夏威夷（对照）')):
        st = {'K': '22107', 'G': '52200', 'H': '51207'}[sid]
        o = pd.read_csv(os.path.join(ROOT, 'processed', f'site_{sid}_reforecast', 'observations.csv'), usecols=['valid_time', 'hs'])
        o.index = pd.to_datetime(o.valid_time.str.replace('Z', ''))
        f = pd.concat([read_gz_tolerant(p)[0] for p in glob.glob(os.path.join(ROOT, 'processed', 'gefs_reforecast', st, '*.csv.gz')) if 2012 <= int(os.path.basename(p)[:4]) <= 2017])
        f['issue_time'] = pd.to_datetime(f.issue_time).dt.tz_localize(None)
        f['valid_time'] = pd.to_datetime(f.valid_time).dt.tz_localize(None)
        f = f.drop_duplicates(['issue_time', 'member', 'valid_time'])
        f = f[(f.valid_time - f.issue_time).dt.total_seconds().between(3 * 3600, 24 * 3600)]
        e = f.groupby(['issue_time', 'valid_time']).hs.mean().reset_index()
        e['obs'] = o.hs.reindex(e.valid_time).to_numpy()
        e = e.dropna()
        pair.append(dict(id=st, name=label, n=len(e), obs=e.obs.mean(), gefs=e.hs.mean(), bias=(e.hs - e.obs).mean(), rmse=np.sqrt(((e.hs - e.obs) ** 2).mean()), r=np.corrcoef(e.hs, e.obs)[0, 1], issues=e.issue_time.nunique(), ok=-1))
    L += [f'| {x["id"]} | {x["name"]} | {x["n"]:,} | {x["obs"]:.2f} | {x["gefs"]:.2f} | {x["bias"]:+.2f} | {x["rmse"]:.2f} | {x["r"]:.3f} | {x["issues"]} | {x["ok"] if x["ok"] >= 0 else "—"} |' for x in pair]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
