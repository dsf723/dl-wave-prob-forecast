"""由 ERA5 区域风场计算站点"上游扇区风"特征（风场输入组 windfield）。

物理思路：站点的浪 = 本地风浪 + 远处涌浪。把站点周围的海面分成 8 个方向扇区（N、NE、E、SE、S、SW、W、NW，
以站点为中心、真北起算）× 2 个距离环（0–300 km 近场、300–1000 km 远场），每块只用海上格点，计算
"吹向站点的风速分量"（风矢量在格点→站点方向上的投影，m/s，负值表示吹离站点）的平均值和最大值：
  wf_<扇区>_<环>_mean、wf_<扇区>_<环>_max，共 8 × 2 × 2 = 32 个特征。
模型同时看到过去 7 天这些特征，能学到远处风生成的涌浪经过若干天才到达站点的滞后关系。

输入：era5_data/windfield_<框>_<年>.nc（download_era5_windfield.py），海陆掩膜 era5_data/lsm_<框>.nc（首次运行自动下载 1 个场）
输出：processed/site_<ID>/windfield_features.csv（逐 3 小时，UTC）
用法：python wind_features.py [A B C K]（东亚框）
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = {s['id']: s for s in json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']}
ERA5 = os.path.join(ROOT, 'era5_data')
BOX_OF = {'A': 'EA', 'B': 'EA', 'C': 'EA', 'K': 'EA', 'G': 'G', 'H': 'H'}
AREA = {'EA': [44, 103, 10, 137], 'G': [24, 134, 3, 155], 'H': [32, -168, 11, -147]}
SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
RINGS = [('near', 0, 300), ('far', 300, 1000)]


def land_sea_mask(box):
    p = os.path.join(ERA5, f'lsm_{box}.nc')
    if not os.path.exists(p):
        import cdsapi
        cdsapi.Client(verify=0, quiet=True).retrieve('reanalysis-era5-single-levels', {
            'product_type': ['reanalysis'], 'variable': ['land_sea_mask'], 'year': ['2020'], 'month': ['01'],
            'day': ['01'], 'time': ['00:00'], 'area': AREA[box], 'grid': [1.0, 1.0],
            'data_format': 'netcdf', 'download_format': 'unarchived'}, p)
    with xr.open_dataset(p) as ds:
        return ds['lsm'].squeeze(drop=True).load()


def geometry(lat0, lon0, lat, lon):
    """站点→格点的距离（km）与方位（度，真北顺时针）；格点→站点的单位方向（东、北分量）。"""
    la0, lo0, la, lo = map(np.radians, (lat0, lon0, lat, lon))
    d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2))
    y = np.sin(lo - lo0) * np.cos(la)
    x = np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(lo - lo0)
    bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
    # 格点→站点方向的方位 ≈ 站点→格点方位 + 180°（1000 km 内足够准确）
    back = np.radians((bearing + 180) % 360)
    return d, bearing, np.sin(back), np.cos(back)


def run(sid):
    site, box = SITES[sid], BOX_OF[sid]
    files = sorted(glob.glob(os.path.join(ERA5, f'windfield_{box}_*.nc')))
    if not files:
        print(f'[跳过] {sid}：没有 windfield_{box}_*.nc')
        return
    lsm = land_sea_mask(box)
    lat0, lon0 = site['point']
    with xr.open_dataset(files[0]) as ds0:
        lat, lon = np.meshgrid(ds0.latitude.values, ds0.longitude.values, indexing='ij')
    sea = lsm.interp(latitude=ds0.latitude, longitude=ds0.longitude, method='nearest').values < 0.5
    d, bearing, ex, ny = geometry(lat0, lon0, lat, lon)
    sector = (((bearing + 22.5) % 360) // 45).astype(int)
    masks = {}
    for k, name in enumerate(SECTORS):
        for ring, r0, r1 in RINGS:
            m = sea & (sector == k) & (d >= r0) & (d < r1)
            if m.sum() >= 2:
                masks[f'wf_{name}_{ring}'] = m
    print(f'[{sid}] {len(masks)} 个扇区块（海上格点 ≥ 2），格点数：' +
          '、'.join(f'{k[3:]}={v.sum()}' for k, v in masks.items()))
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            u, v = ds['u10'].values, ds['v10'].values               # (time, lat, lon)
            toward = u * ex + v * ny                                # 吹向站点的分量
            cols = {}
            for k, m in masks.items():
                x = toward[:, m]
                cols[k + '_mean'] = x.mean(1)
                cols[k + '_max'] = x.max(1)
            parts.append(pd.DataFrame(cols, index=pd.DatetimeIndex(ds['valid_time'].values, name='valid_time')))
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated()].astype('float32').round(3)
    p = os.path.join(ROOT, 'processed', f'site_{sid}', 'windfield_features.csv')
    out.index = out.index.strftime('%Y-%m-%dT%H:%M:%SZ')
    out.to_csv(p)
    print(f'[完成] {sid}：{len(out)} 个时刻 × {out.shape[1]} 个特征，{out.index[0]} – {out.index[-1]} → {p}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings
    warnings.filterwarnings('ignore')
    for sid in (sys.argv[1:] or ['A']):
        run(sid)
