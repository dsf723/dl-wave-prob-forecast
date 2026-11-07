"""【临时备用】经 Open-Meteo 获取各站点 ERA5 逐时序列（1995–2024），生成与 CDS 版相同格式的 observations。

为什么有这个脚本：CDS 下载被限流、排队很慢；Open-Meteo 提供同一套 ERA5（海浪 era5_ocean、大气 era5），几分钟即可取完，
用来先把整条流程在真实数据上跑通、做预览分析。论文正式结果应使用 CDS 下载的数据（processed/site_X）。
输出到 processed_openmeteo/site_X/（与 CDS 版分开，避免混用）：observations.csv（逐 3 小时）、observations_hourly.csv、manifest.json。
注意：
  - Open-Meteo 取最近的有效海上格点（cell_selection=sea），不插值；实际格点写入 manifest；
  - 海浪周期 wave_period 推测对应 ERA5 mwp（能量周期），wave_peak_period 对应 pp1d，均未逐项核实；
  - 气温由 °C 换算为 K，气压由 hPa 换算为 Pa；风由风速风向换算为 u10、v10。
用法：python fetch_era5_openmeteo.py [A B C G K H]
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
OUT = os.path.join(ROOT, 'processed_openmeteo')
CHUNKS = [(1995, 2004), (2005, 2014), (2015, 2024)]


def get(url):
    for k in range(5):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research'}), timeout=600).read())
        except Exception as e:
            print(f'    重试 {k + 1}：{str(e)[:120]}', flush=True)
            time.sleep(60 * (k + 1))
    raise RuntimeError('Open-Meteo 多次失败')


def fetch(site):
    lat, lon = site['point']
    wave, atm, cells = [], [], {}
    for y0, y1 in CHUNKS:
        w = get(f'https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&start_date={y0}-01-01&end_date={y1}-12-31'
                '&hourly=wave_height,wave_period,wave_direction,wave_peak_period&models=era5_ocean&timezone=GMT&cell_selection=sea')
        cells['wave'] = (w['latitude'], w['longitude'])
        wave.append(pd.DataFrame(w['hourly']))
        a = get(f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={y0}-01-01&end_date={y1}-12-31'
                '&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,pressure_msl&models=era5&wind_speed_unit=ms&timezone=GMT&cell_selection=sea')
        cells['atm'] = (a['latitude'], a['longitude'])
        atm.append(pd.DataFrame(a['hourly']))
        time.sleep(2)
    w = pd.concat(wave).set_index('time')
    a = pd.concat(atm).set_index('time')
    df = w.join(a, how='outer')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated()].sort_index()
    rad = np.deg2rad(df['wind_direction_10m'])
    out = pd.DataFrame(index=df.index)
    out['site_id'] = site['id']
    out['valid_time'] = df.index.strftime('%Y-%m-%dT%H:%M:%SZ')
    out['available_time'] = (df.index + pd.Timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    out['hs'] = df['wave_height']
    out['te'] = df['wave_period']
    out['mwd'] = df['wave_direction']
    out['u10'] = -df['wind_speed_10m'] * np.sin(rad)          # 风向为来向
    out['v10'] = -df['wind_speed_10m'] * np.cos(rad)
    out['t2m'] = df['temperature_2m'] + 273.15
    out['msl'] = df['pressure_msl'] * 100
    out['tp'] = df['wave_peak_period']
    out['tm02'] = np.nan
    tc = os.path.join(ROOT, 'processed', 'tracks', f"tc_hourly_{site['id']}.csv")
    if os.path.exists(tc):
        lab = pd.read_csv(tc, usecols=['valid_time', 'label_tc_distance_km', 'event_id'], index_col=0, parse_dates=True)
        out['label_tc_distance_km'] = lab['label_tc_distance_km'].reindex(df.index).values
        out['event_id'] = lab['event_id'].reindex(df.index).fillna('').values
    d = os.path.join(OUT, f"site_{site['id']}")
    os.makedirs(d, exist_ok=True)
    out.to_csv(os.path.join(d, 'observations_hourly.csv'), index=False)
    out[out.index.hour % 3 == 0].to_csv(os.path.join(d, 'observations.csv'), index=False)
    miss = {c: round(float(out[c].isna().mean()) * 100, 2) for c in ('hs', 'te', 'mwd', 'u10', 't2m', 'msl')}
    json.dump({'mode': 'reanalysis_proxy', 'availability_verified': False, 'site_id': site['id'], 'site_lat': lat, 'site_lon': lon,
               'observation_source': 'ERA5 via Open-Meteo（海浪 era5_ocean，大气 era5）——临时备用，正式结果用 CDS 版',
               'grid_cells': cells, 'period_definition': 'te = Open-Meteo wave_period（推测为 ERA5 mwp，未核实）；tp = wave_peak_period',
               'availability_provenance': 'available_time = valid_time + 5 天（近似）', 'missing_percent': miss,
               'downloaded_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'notes': '仅用于流程联调与预览'},
              open(os.path.join(d, 'manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[完成] {site['id']} {site['label']}：{len(out)} 个逐时时次，格点 {cells}，缺测 {miss}", flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sel = set(sys.argv[1:])
    for s in sorted(SITES, key=lambda s: s['priority']):
        if not sel or s['id'] in sel:
            fetch(s)
