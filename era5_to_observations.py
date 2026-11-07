"""
将 era5_data/ 下各站点的 ERA5 文件（zip 或 netCDF；按季度、按年、上下半年、补充变量 *_extra 均可）
转换为 processed/site_X/observations.csv（逐 3 小时，供 outputs/wave_forecast 使用）
和 observations_hourly.csv（逐时），并写 manifest.json。站点见 metadata/sites.json。

处理要点：
- 站点序列：双线性插值；周围格点含陆地（NaN）时改取最近的有效海点，并记录该点坐标与距离。
- 波向 mwd 先拆成 sin/cos 分别插值再合成，不直接线性插值角度。
- 周期：mwp = 能量周期 Te（ECMWF 定义 m-1/m0）；pp1d = 谱峰周期 Tp；mp2 = 跨零周期 Tm02（与浮标 APD 对应）。
- 台风标签列 label_tc_distance_km、event_id 先留空，由后续脚本用 IBTrACS 填写。
用法：python era5_to_observations.py [A B C G K H]
"""
import glob
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
ERA5_DIR = os.path.join(ROOT, 'era5_data')
OUT_BASE = os.path.join(ROOT, 'processed')
VARS = {'swh': 'hs', 'mwp': 'te', 'pp1d': 'tp', 'mp2': 'tm02', 'u10': 'u10', 'v10': 'v10', 't2m': 't2m', 'msl': 'msl'}


def _nc_paths(path, tmpdir):
    """zip（CDS 多数据流时返回）则解压，返回其中的 .nc；否则返回自身。"""
    with open(path, 'rb') as f:
        is_zip = f.read(2) == b'PK'
    if not is_zip:
        return [path]
    with zipfile.ZipFile(path) as z:
        sub = os.path.join(tmpdir, os.path.basename(path))
        z.extractall(sub)
    return glob.glob(os.path.join(sub, '*.nc'))


def _standardize(ds, lon_site):
    if 'valid_time' not in ds.coords and 'time' in ds.coords:
        ds = ds.rename({'time': 'valid_time'})
    for d in ('number', 'expver'):
        if d in ds.dims:
            ds = ds.isel({d: 0})
    ds = ds.drop_vars([c for c in ('number', 'expver') if c in ds.coords], errors='ignore')
    if 'longitude' in ds.dims and lon_site < 0 and float(ds.longitude.max()) > 180:     # 0–360 → -180–180
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby('longitude')
    return ds


def _nearest_valid(da, lat, lon):
    first = da.isel(valid_time=0)
    la, lo = np.meshgrid(da.latitude.values, da.longitude.values, indexing='ij')
    d = np.where(np.isfinite(first.values), np.hypot((lo - lon) * np.cos(np.radians(lat)), la - lat), np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    km = float(d[i, j]) * 111.2
    return da.isel(latitude=i, longitude=j), (float(da.latitude[i]), float(da.longitude[j]), round(km, 1))


def _point(da, lat, lon):
    if 'latitude' not in da.dims:                         # 单点时间序列文件：已是 CDS 选好的最近格点
        cl = (float(da['latitude']) if 'latitude' in da.coords else lat, float(da['longitude']) if 'longitude' in da.coords else lon)
        km = round(float(np.hypot((cl[1] - lon) * np.cos(np.radians(lat)), cl[0] - lat)) * 111.2, 1)
        return da, (cl[0], cl[1], km)
    v = da.interp(latitude=lat, longitude=lon)
    if np.isfinite(v.isel(valid_time=0)):
        return v, (lat, lon, 0.0)
    return _nearest_valid(da, lat, lon)


def load_site_file(path, lat, lon, tmpdir, used):
    frames = []
    for nc in _nc_paths(path, tmpdir):
        with xr.open_dataset(nc) as ds0:
            ds = _standardize(ds0, lon).load()
        cols = {}
        for k, v in VARS.items():
            if k in ds:
                cols[v], used[v] = _point(ds[k], lat, lon)
        if 'mwd' in ds:
            rad = np.deg2rad(ds['mwd'])
            s, used['mwd'] = _point(np.sin(rad), lat, lon)
            c, _ = _point(np.cos(rad), lat, lon)
            cols['mwd'] = (np.rad2deg(np.arctan2(s, c)) + 360) % 360
        if cols:
            frames.append(pd.DataFrame({k: v.values for k, v in cols.items()},
                                       index=pd.DatetimeIndex(ds['valid_time'].values, name='valid_time')))
    return frames


def process_site(site):
    lat, lon = site['point']
    files = sorted(glob.glob(os.path.join(ERA5_DIR, f"{site['name']}_*.nc")) +      # 网格文件（季度/年/补充变量）
                   glob.glob(os.path.join(ERA5_DIR, f"ts_{site['name']}_*.nc")))     # 单点时间序列文件
    if not files:
        print(f"[跳过] 站点 {site['id']}（{site['name']}）还没有 ERA5 文件")
        return
    print(f"[处理] 站点 {site['id']} {site['label']}：{len(files)} 个文件")
    used, parts = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            try:
                parts += load_site_file(f, lat, lon, tmp, used)
            except Exception as e:
                print(f"    [警告] {os.path.basename(f)}：{e}")
    if not parts:
        print(f"[失败] 站点 {site['id']} 无有效数据")
        return
    # 同一时刻可能来自不同文件（季度文件 + 补充变量文件）：按时间合并各列
    df = pd.concat(parts).groupby(level=0).first().sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    full = pd.date_range(df.index.min(), df.index.max(), freq='1h')
    df = df.reindex(full)
    df.index.name = 'valid_time'
    out = pd.DataFrame(index=df.index)
    out['site_id'] = site['id']
    out['valid_time'] = df.index.strftime('%Y-%m-%dT%H:%M:%SZ')
    out['available_time'] = (df.index + pd.Timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ')  # ERA5T 约滞后 5 天
    for c in ('hs', 'te', 'mwd', 'u10', 'v10', 't2m', 'msl', 'tp', 'tm02'):
        out[c] = df[c].values if c in df else np.nan
    out['label_tc_distance_km'] = np.nan
    out['event_id'] = ''
    tc = os.path.join(OUT_BASE, 'tracks', f"tc_hourly_{site['id']}.csv")   # 若已运行 label_typhoons.py，直接带上台风标签
    if os.path.exists(tc):
        lab = pd.read_csv(tc, usecols=['valid_time', 'label_tc_distance_km', 'event_id'], index_col=0, parse_dates=True)
        out['label_tc_distance_km'] = lab['label_tc_distance_km'].reindex(df.index).values
        out['event_id'] = lab['event_id'].reindex(df.index).fillna('').values
    d = os.path.join(OUT_BASE, f"site_{site['id']}")
    os.makedirs(d, exist_ok=True)
    out.to_csv(os.path.join(d, 'observations_hourly.csv'), index=False)
    three = out[out.index.hour % 3 == 0]
    three.to_csv(os.path.join(d, 'observations.csv'), index=False)
    years = sorted(set(out.index.year))
    miss = {c: round(float(out[c].isna().mean()) * 100, 2) for c in ('hs', 'te', 'mwd', 'u10', 't2m', 'msl', 'tp', 'tm02')}
    manifest = {
        'mode': 'reanalysis_proxy', 'availability_verified': False, 'site_id': site['id'],
        'site_lat': lat, 'site_lon': lon, 'site_label': site['label'],
        'observation_source': 'ERA5 hourly single levels (final reanalysis), Copernicus CDS',
        'forecast_source': None,
        'period_definition': 'te = mwp（ECMWF：基于 m-1 阶矩的平均周期，即能量周期）；tp = pp1d 谱峰周期；tm02 = mp2 跨零周期',
        'availability_provenance': 'available_time = valid_time + 5 天（ERA5T 发布滞后的近似，未逐条核实）',
        'track_source': 'label_tc_distance_km / event_id 待由 IBTrACS 最佳路径填写（仅诊断用）',
        'units_and_direction': 'hs m; te/tp/tm02 s; mwd 来向，真北 0°顺时针; u10/v10 m/s; t2m K; msl Pa',
        'extraction': {k: {'lat': v[0], 'lon': v[1], 'distance_km': v[2]} for k, v in used.items()},
        'coverage': f'{years[0]}–{years[-1]}，共 {len(out)} 个逐时时次',
        'missing_percent': miss,
        'downloaded_at': '见 era5_data/download_log.txt',
        'notes': '缺季度/缺变量的时段为空值，未做插补；近岸格点误差未校正',
    }
    json.dump(manifest, open(os.path.join(d, 'manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[完成] {d}：逐时 {len(out)} 行，逐 3 小时 {len(three)} 行，{years[0]}–{years[-1]}")
    print(f"    缺测（%）：{miss}")
    print(f"    提取位置：{manifest['extraction'].get('hs')}")


if __name__ == '__main__':
    sel = set(sys.argv[1:])
    for s in sorted(SITES, key=lambda s: s['priority']):
        if not sel or s['id'] in sel:
            process_site(s)
