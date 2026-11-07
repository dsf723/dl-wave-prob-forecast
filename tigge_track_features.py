"""TIGGE 台风路径预报 → 模型的"未来"输入特征（每个起报 × 每个预见时刻），生成带台风特征的训练数据集。

用哪一次路径预报（不偷看）：决策时刻 = 起报 + 6 h（与 GEFS 可得时间一致）；只用 base_time ≤ 决策时刻 − 12 h 的
  ECMWF 路径预报（00 UTC 起报时就是前一天 12 UTC，缺时为前一天 00 UTC），且不早于决策时刻 − 48 h，否则记为缺失。
特征（预见时刻 t；ECMWF 集合 51 个成员 0–50，−1 为确定性预报；路径每 6 h 一个点，按时间线性插值到 t）：
  tc_p500 / tc_p1000   有气旋中心在站点 500 / 1000 km 内的成员比例
  tc_dist_med_km       各成员"最近气旋距离"的中位数（没有气旋记 3000 km，最大取 3000）
  tc_vmax_near_ms      1000 km 内最近气旋的最大风速，成员平均（没有记 0）
  tc_det_dist_km       确定性预报的最近气旋距离（同样封顶 3000 km）
缺失（取 NaN，模型输入会自动附带"缺测"标记）：没有可用的路径预报（TIGGE 2006-10 前，或存档缺那几天），
  或 t 超出路径预报时效（240 h，约到第 9.5 天）。存档里有文件但太平洋没有气旋 → 特征取"无台风"值（不是缺失）。
输出：processed/site_<ID>_reforecast_tc/（observations.csv、manifest.json 复制自 site_<ID>_reforecast，
  forecasts.csv 多上面 5 列）
用法：python tigge_track_features.py [G K H] [--suffix _daily]     （--suffix：输入 site_<ID>_reforecast<后缀>，输出再加 _tc）
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = {s['id']: s for s in json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']}
TRACKS = os.path.join(ROOT, 'tigge_tracks', 'ecmf_tracks_pacific.csv.gz')
DECISION_H, CUTOFF_H, OLDEST_H, CAP_KM = 6, 12, 48, 3000.0
COLS = ['tc_p500', 'tc_p1000', 'tc_dist_med_km', 'tc_vmax_near_ms', 'tc_det_dist_km']


RUNS_FILE = os.path.join(ROOT, 'tigge_tracks', 'available_runs.txt')


def available_runs():
    """Base times of every downloaded ECMWF track file (a file with no Pacific storm is still a valid run).
    The list is also saved to tigge_tracks/available_runs.txt so the server (without the 12 GB of XML) can use it."""
    names = glob.glob(os.path.join(ROOT, 'tigge_tracks', 'ecmf', '*', 'z_tigge_c_ecmf_*.xml'))
    if names:
        stamps = sorted({os.path.basename(n)[15:25] for n in names})                 # YYYYMMDDHH
        open(RUNS_FILE, 'w', encoding='utf-8').write(' '.join(stamps) + ' ')
    else:
        stamps = open(RUNS_FILE, encoding='utf-8').read().split()
    return pd.DatetimeIndex(pd.to_datetime(stamps, format='%Y%m%d%H', utc=True))


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def run_tracks(tracks):
    """{base_time: [(member, lead_h array, lat, lon, vmax), ...]} with duplicate fixes averaged."""
    out = {}
    g = tracks.groupby(['base_time', 'member', 'storm_id', 'lead_h'], sort=True)[['lat', 'lon', 'vmax_ms']].mean().reset_index()
    for (base, member, _), t in g.groupby(['base_time', 'member', 'storm_id'], sort=False):
        out.setdefault(base, []).append((member, t.lead_h.to_numpy(float), t.lat.to_numpy(), t.lon.to_numpy(),
                                         t.vmax_ms.fillna(0).to_numpy()))
    return out


def features(tracks_of_run, lead, site_lat, site_lon):
    """Five features for the lead times `lead` (hours after base) of one run; NaN beyond 240 h."""
    n = len(lead)
    dist = np.full((51, n), CAP_KM); vmax = np.zeros((51, n)); det = np.full(n, CAP_KM)
    for member, lh, la, lo, vm in tracks_of_run:
        inside = (lead >= lh[0]) & (lead <= lh[-1])
        if not inside.any():
            continue
        x = lead[inside]
        lat = np.interp(x, lh, la)
        lon = np.interp(x, lh, np.unwrap(lo, period=360))
        d = np.minimum(distance_km(site_lat, site_lon, lat, lon), CAP_KM)
        v = np.interp(x, lh, vm)
        if member < 0:
            det[inside] = np.minimum(det[inside], d)
            continue
        closer = d < dist[member, inside]
        idx = np.flatnonzero(inside)[closer]
        dist[member, idx], vmax[member, idx] = d[closer], v[closer]
    out = np.column_stack([(dist <= 500).mean(0), (dist <= 1000).mean(0), np.median(dist, 0),
                           np.where(dist <= 1000, vmax, 0).mean(0), det])
    out[lead > 240] = np.nan
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    runs = available_runs()
    print(f'可用的 ECMWF 路径预报 {len(runs)} 次（{runs[0]:%Y-%m-%d} 至 {runs[-1]:%Y-%m-%d}）', flush=True)
    tracks = pd.read_csv(TRACKS)
    tracks['base_time'] = pd.to_datetime(tracks.base_time, utc=True)
    by_run = run_tracks(tracks)
    del tracks
    args = sys.argv[1:]
    suffix = ''
    if '--suffix' in args:
        i = args.index('--suffix'); suffix = args[i + 1]; del args[i:i + 2]
    for sid in (args or ['G', 'K', 'H']):
        site = SITES[sid]
        lat, lon = site['buoy']['lat'], site['buoy']['lon'] % 360
        src = os.path.join(ROOT, 'processed', f'site_{sid}_reforecast{suffix}')
        dst = src + '_tc'
        os.makedirs(dst, exist_ok=True)
        fc = pd.read_csv(os.path.join(src, 'forecasts.csv'))
        issue, valid = pd.to_datetime(fc.issue_time, utc=True), pd.to_datetime(fc.valid_time, utc=True)
        feats = np.full((len(fc), len(COLS)), np.nan)
        n_used = n_missing = 0
        for it, rows in fc.groupby(issue, sort=True).indices.items():
            cutoff = it + pd.Timedelta(hours=DECISION_H - CUTOFF_H)
            ok = runs[(runs <= cutoff) & (runs >= cutoff - pd.Timedelta(hours=OLDEST_H - CUTOFF_H))]
            if len(ok) == 0:
                n_missing += 1
                continue
            base = ok[-1]
            lead = ((valid.iloc[rows] - base) / pd.Timedelta(hours=1)).to_numpy(float)
            feats[rows] = features(by_run.get(base, []), lead, lat, lon)
            n_used += 1
        for j, c in enumerate(COLS):
            fc[c] = np.round(feats[:, j], 4)
        fc.to_csv(os.path.join(dst, 'forecasts.csv'), index=False)
        for f in ('observations.csv', 'manifest.json'):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        hot = fc[fc.tc_p500 > 0]
        print(f'[完成] {sid}：{n_used} 次起报有路径预报，{n_missing} 次没有（缺失）；'
              f'有成员预报 500 km 内有气旋的时刻 {len(hot)} 个（{hot.issue_time.nunique()} 次起报）→ {dst}', flush=True)


if __name__ == '__main__':
    main()
