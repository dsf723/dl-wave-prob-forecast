"""试提：从 GEFS 回算全球 0.25° 网格取舟山点的波高，确认这个格点可用（约 52 MB）。

资料：noaa-nws-gefswaves-reforecast-pds，GEFSv12/reforecast/2019/20190102/gridded/gefs.wave.20190102.c00.global.0p25.grib2 + .idx；
      按 .idx 只读 HTSGW 的 surface 记录（时效 3–384 h，每 3 小时一条），在内存里解码，不保存原始文件。
注意（11-02 试提发现）：网格文件的 "N hour fcst" 从 d= 的 03 时算起，不是从 00 时起报算起，anl 对应时效 3 h。第一次运行按 00 时算，比站点序列晚了 3 小时
      （22107、C6W08 相差 +3 小时后 RMSE 0.009、0.004 m），已改为按 d= 算有效时刻。以后全量提取必须按 d= 算。
核对：①舟山点四周 0.25° 格点是否为海上（第一个时效）；②同一张场上取 3 个已有站点序列的点（22107、46757B、C6W08），与本机 tab.nc 提取的同一起报、
      同一成员比较，确认时间对齐和取点方法；③舟山点与 ERA5 A 点同时刻比较；④这段时间 55 km 内的 CFOSAT 过境。
输出：results/zhoushan_gefs_probe/report.md、hs_20190102_c00.csv
用法：python probe_zhoushan_gefs_grid.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import requests

import compare_era5_altimeter as cea
from download_gefs_operational import decode_at, fetch
from prepare_reforecast_dataset import read_gz_tolerant

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'results', 'zhoushan_gefs_probe')
BASE = 'https://noaa-nws-gefswaves-reforecast-pds.s3.amazonaws.com/GEFSv12/reforecast/2019/20190102/gridded/gefs.wave.20190102.c00.global.0p25'
ISSUE = pd.Timestamp('2019-01-02')
A = (30.5, 123.0)
CHECK = {'22107': (33.08, 126.03), '46757B': (24.76, 120.84), 'C6W08': (26.38, 120.54)}
CORNERS = [(30.5 + a, 123.0 + b) for a in (-.25, 0, .25) for b in (-.25, 0, .25)]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs(OUT, exist_ok=True)
    ses = requests.Session()
    rows = [ln.split(':') for ln in fetch(ses, BASE + '.idx').decode().strip().splitlines()]
    pts = [A] + list(CHECK.values()) + CORNERS
    res, nbytes, how0 = [], 0, None
    for i, r in enumerate(rows):
        if r[3] != 'HTSGW' or r[4] != 'surface':
            continue
        h = 3 + (0 if r[5] == 'anl' else int(r[5].split()[0]))          # 真实时效 = 03 时 + N
        if h > 384:
            continue
        rng = (int(r[1]), int(rows[i + 1][1]) - 1)
        msg = fetch(ses, BASE + '.grib2', rng)
        nbytes += len(msg)
        v, how = decode_at(msg, pts)
        how0 = how0 or how
        res.append([h] + list(v))
    names = ['lead_h', 'A'] + list(CHECK) + [f'g{i}' for i in range(len(CORNERS))]
    d = pd.DataFrame(res, columns=names)
    # 第一个时效检查 d= 字段：回算的 anl 记为 03 时，确认 lead 的零点，见下面与 tab.nc 的比较
    ref_time = rows[0][2][2:]
    L = ['# 舟山点 GEFS 回算网格试提（2019-01-02，c00，只取波高）', '', f'读取 {len(d)} 条记录，共 {nbytes / 1e6:.1f} MB。索引里的 d= 为 {ref_time}。', '',
         '## 舟山点四周格点（缺测 = 陆地）', '', f'第一个时效（{int(d.lead_h[0])} h）波高，北在上：', '', '| 纬度 \\ 经度 | 122.75 | 123.00 | 123.25 |', '|---:|---:|---:|---:|']
    for ia, lat in zip((6, 3, 0), (30.75, 30.5, 30.25)):
        L.append(f'| {lat} | ' + ' | '.join('缺测' if np.isnan(d[f"g{ia + ib}"][0]) else f'{d[f"g{ia + ib}"][0]:.2f}' for ib in range(3)) + ' |')
    L += ['', f'舟山点取值方式代码 {how0[0]}（0 = 四角都有值）；全部时效缺测 {int(d.A.isna().sum())} 条。', '', '## 同一张场与本机站点序列比较（同起报、同成员）', '',
          '| 站号 | 配对 | 相差 0 小时 RMSE | 相差 +3 小时 RMSE | 相差 −3 小时 RMSE | 平均差（相差 0）|', '|---|---:|---:|---:|---:|---:|']
    for sid in CHECK:
        f = read_gz_tolerant(os.path.join(ROOT, 'processed', 'gefs_reforecast', sid, '2019.csv.gz'))[0]
        f['issue_time'] = pd.to_datetime(f.issue_time).dt.tz_localize(None)
        f['valid_time'] = pd.to_datetime(f.valid_time).dt.tz_localize(None)
        f = f[(f.issue_time == ISSUE) & (f.member == 'c00')].drop_duplicates('valid_time').set_index('valid_time').hs
        rm = {}
        for sh in (0, 3, -3):
            t = ISSUE + pd.to_timedelta(d.lead_h + sh, unit='h')
            x = pd.DataFrame(dict(g=d[sid].to_numpy(), s=f.reindex(t).to_numpy())).dropna()
            rm[sh] = (len(x), np.sqrt(((x.g - x.s) ** 2).mean()), (x.g - x.s).mean())
        L.append(f'| {sid} | {rm[0][0]} | {rm[0][1]:.3f} | {rm[3][1]:.3f} | {rm[-3][1]:.3f} | {rm[0][2]:+.3f} |')
    e = pd.read_csv(os.path.join(ROOT, 'processed', 'site_A', 'observations_hourly.csv'), usecols=['valid_time', 'hs'])
    e.index = pd.to_datetime(e.valid_time.str.replace('Z', ''))
    t = ISSUE + pd.to_timedelta(d.lead_h, unit='h')
    d['valid_time'] = t
    d['era5'] = e.hs.reindex(t).to_numpy()
    L += ['', '## 舟山点与 ERA5 A 点同时刻', '', '| 时效段 | GEFS c00 平均 | ERA5 平均 | 偏差 | RMSE | 相关 |', '|---|---:|---:|---:|---:|---:|']
    for lo, hi in ((3, 72), (75, 168), (171, 384)):
        g = d[d.lead_h.between(lo, hi)]
        L.append(f'| {lo}–{hi} h | {g.A.mean():.2f} | {g.era5.mean():.2f} | {(g.A - g.era5).mean():+.2f} | {np.sqrt(((g.A - g.era5) ** 2).mean()):.2f} | {np.corrcoef(g.A, g.era5)[0, 1]:.2f} |')
    c = pd.read_csv(os.path.join(ROOT, '验证数据', 'CFOSAT_L3_SWH_VAVH_Zhoushan_2019-2020.csv'), comment='#')
    c['time'] = pd.to_datetime(c.time.str.replace('Z', '').str.replace('.000', ''))
    c['dist'] = cea.haversine(A[0], A[1], c.latitude, c.longitude)
    c = c[(c.dist <= 55) & c.time.between(t.min(), t.max())].sort_values('time')
    c['pid'] = (c.time.diff() > pd.Timedelta('60s')).cumsum()
    p = c.groupby('pid').agg(time=('time', 'mean'), n=('value', 'size'), hs=('value', 'median'))
    p = p[p.n >= 3]
    L += ['', '## 这段时间 55 km 内的 CFOSAT 过境', '', '| 时刻 | CFOSAT | GEFS c00（插值）| ERA5 |', '|---|---:|---:|---:|']
    x = t.values.astype('datetime64[s]').astype(np.int64)
    for q in p.itertuples():
        ts = np.datetime64(q.time, 's').astype(np.int64)
        L.append(f'| {q.time:%Y-%m-%d %H:%M} | {q.hs:.2f} | {np.interp(ts, x, d.A):.2f} | {np.interp(ts, x, d.era5):.2f} |')
    if p.empty:
        L.append('| 无 | | | |')
    d[['valid_time', 'lead_h', 'A', 'era5']].rename(columns={'A': 'hs_gefs_c00', 'era5': 'hs_era5'}).round(3).to_csv(os.path.join(OUT, 'hs_20190102_c00.csv'), index=False)
    open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
