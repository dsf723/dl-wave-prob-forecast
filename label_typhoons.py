"""用 IBTrACS 最佳路径给各站点逐时标注最近的热带气旋（事后诊断标签，不能作为模型输入）。

输入：raw/tracks/ibtracs.since1980.list.v04r01.csv（NOAA NCEI，v04r01，1980 年以来全部海区）
输出：
  processed/tracks/tc_hourly_<站点ID>.csv   逐时：最近气旋的距离、方位、强度、移速、SID、名称
  并回填 processed/site_X/observations*.csv 与 processed/buoy/<ID>_*.csv 的 label_tc_distance_km、event_id 两列
约定（与 outputs/wave_forecast/DATA_SCHEMA.md 一致）：
  - 当时 IBTrACS 中没有任何活跃气旋 → 距离记 20000 km（表示"确认无近距离影响"）；
  - event_id = 气旋 SID，仅当距离 ≤ EVENT_RADIUS_KM 时填写，诊断时可再按 300/500/800 km 细分；
  - 强度优先用 USA_WIND（JTWC/NHC，1 分钟平均，kt），缺失时用 WMO_WIND（西北太平洋为日本气象厅 10 分钟平均）；
    两者平均时段不同，tc_wind_source 列注明来源；换算 1 kt = 0.5144 m/s；
  - 只用 TRACK_TYPE 为 main / PROVISIONAL 的记录，路径线性插值到逐时（经度先展开，跨日界线不出错）；
  - 只考虑 WP（含南海）与 EP（含中太平洋，夏威夷用）两个海区。
寒潮事件需要气温资料，待 ERA5 气温下载后另行标注。
用法：python label_typhoons.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
TRACKS = os.path.join(ROOT, 'raw', 'tracks', 'ibtracs.since1980.list.v04r01.csv')
OUT = os.path.join(ROOT, 'processed', 'tracks')
EVENT_RADIUS_KM = 800
KT = 0.514444


def _fill_blank(a):
    """空字符串用前一个（开头用后一个）非空值填补。"""
    a = np.array(a, dtype=object)
    good = [i for i, v in enumerate(a) if v]
    if not good:
        return a
    idx = np.maximum.accumulate(np.where(a != '', np.arange(len(a)), -1))
    idx[idx < 0] = good[0]
    return a[idx]


def load_tracks():
    cols = ['SID', 'NAME', 'BASIN', 'ISO_TIME', 'TRACK_TYPE', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'STORM_SPEED']
    d = pd.read_csv(TRACKS, skiprows=[1], usecols=cols, low_memory=False, na_values=[' ', ''])
    d = d[d['BASIN'].isin(['WP', 'EP']) & d['TRACK_TYPE'].str.lower().isin(['main', 'provisional'])]
    d['ISO_TIME'] = pd.to_datetime(d['ISO_TIME'])
    for c in ('LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'STORM_SPEED'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['LAT', 'LON'])
    rows = []
    for sid, g in d.groupby('SID'):
        g = g.sort_values('ISO_TIME').drop_duplicates('ISO_TIME')
        if len(g) < 2:
            continue
        t = g['ISO_TIME'].values.astype('datetime64[h]').astype('int64')
        th = np.arange(t[0], t[-1] + 1)
        lon = np.rad2deg(np.unwrap(np.deg2rad(g['LON'].values % 360)))
        wind = g['USA_WIND'].where(g['USA_WIND'].notna(), g['WMO_WIND'])
        src = np.where(g['USA_WIND'].notna(), 'USA_1min', np.where(g['WMO_WIND'].notna(), 'WMO', ''))
        ok = wind.notna().values
        rows.append(pd.DataFrame({
            'time': th, 'sid': sid, 'name': g['NAME'].iloc[0],
            'lat': np.interp(th, t, g['LAT'].values), 'lon': np.interp(th, t, lon) % 360,
            'vmax_ms': np.interp(th, t[ok], wind.values[ok] * KT) if ok.sum() >= 2 else np.nan,
            'speed_ms': np.interp(th, t, g['STORM_SPEED'].fillna(g['STORM_SPEED'].median()).values * KT),
            'wind_src': _fill_blank(src)[np.clip(np.searchsorted(t, th, 'right') - 1, 0, len(t) - 1)],
        }))
    tr = pd.concat(rows, ignore_index=True)
    tr['time'] = pd.to_datetime(tr['time'].values.astype('datetime64[h]'))
    return tr


def nearest_for_site(tr, lat, lon):
    la1, lo1 = np.deg2rad(lat), np.deg2rad(lon % 360)
    la2, lo2 = np.deg2rad(tr['lat'].values), np.deg2rad(tr['lon'].values)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    brg = np.arctan2(np.sin(lo2 - lo1) * np.cos(la2), np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1))
    x = tr.assign(dist=dist, bsin=np.sin(brg), bcos=np.cos(brg))
    x = x.sort_values('dist').drop_duplicates('time').sort_values('time').set_index('time')
    full = pd.date_range('1980-01-01', x.index.max(), freq='1h')
    out = pd.DataFrame(index=full)
    out['label_tc_distance_km'] = x['dist'].reindex(full).fillna(20000.0).round(1)
    for c, n in (('bsin', 'tc_bearing_sin'), ('bcos', 'tc_bearing_cos'), ('vmax_ms', 'tc_vmax_ms'),
                 ('speed_ms', 'tc_speed_ms'), ('sid', 'tc_sid'), ('name', 'tc_name'), ('wind_src', 'tc_wind_source')):
        out[n] = x[c].reindex(full)
    out['event_id'] = np.where(out['label_tc_distance_km'] <= EVENT_RADIUS_KM, out['tc_sid'], '')
    out.index.name = 'valid_time'
    return out


def fill_labels(path, lab):
    df = pd.read_csv(path, low_memory=False)
    t = pd.to_datetime(df['valid_time'].str.replace('Z', ''), errors='coerce').dt.floor('h')
    df['label_tc_distance_km'] = lab['label_tc_distance_km'].reindex(t).values
    df['event_id'] = lab['event_id'].reindex(t).fillna('').values
    df.to_csv(path, index=False)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = load_tracks()
    print(f'IBTrACS：{tr["sid"].nunique()} 个气旋，逐时位置 {len(tr)} 条，{tr["time"].min():%Y-%m} – {tr["time"].max():%Y-%m-%d}')
    for s in SITES:
        lat, lon = s['point']
        lab = nearest_for_site(tr, lat, lon)
        lab.to_csv(os.path.join(OUT, f"tc_hourly_{s['id']}.csv"))
        sub = lab.loc['1995':'2024']
        n_ev = sub.loc[sub['event_id'] != '', 'event_id'].nunique()
        n500 = sub.loc[sub['label_tc_distance_km'] <= 500, 'tc_sid'].nunique()
        n300 = sub.loc[sub['label_tc_distance_km'] <= 300, 'tc_sid'].nunique()
        print(f"[{s['id']}] 1995–2024：800 km 内 {n_ev} 个气旋，500 km 内 {n500} 个，300 km 内 {n300} 个；"
              f"500 km 内时长 {(sub['label_tc_distance_km'] <= 500).sum()} 小时")
        targets = glob.glob(os.path.join(ROOT, 'processed', f"site_{s['id']}", 'observations*.csv')) + \
            glob.glob(os.path.join(ROOT, 'processed', 'buoy', f"{s['id']}_*.csv"))
        for p in targets:
            df = fill_labels(p, lab)
            print(f"    回填 {os.path.relpath(p, ROOT)}：{(df['event_id'] != '').sum()} 行属于气旋事件")


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
