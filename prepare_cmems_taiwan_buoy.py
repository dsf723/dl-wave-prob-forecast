"""Copernicus 台湾海域浮标 → processed/buoy/TW<站号>_<站号>.csv（与 NDBC、KMA 浮标同一格式），并把站点登记到 metadata/sites.json。

输入：raw/insitu_taiwan_cmems/GL_TS_MO_<站号>.nc（INSITU_GLO_WAV_DISCRETE_MY_013_045，逐时，CC BY 4.0，只有波高 VGHS）。
质控：只用 VGHS_QC = 1，hs 限 0–25 m，重复时刻取第一条。没有周期、波向，这些列留空。
站点：strait = 台湾海峡内 5 站（C6W08 马祖、46757B 新竹、46735A 澎湖、46787A 金门、46778A 七股）；其余 9 站也转换，先不进海峡模型。
台风标签：只给 TW 站算（label_typhoons.py 的 main 会重写所有站的文件，这里不调用它）。
用法：python prepare_cmems_taiwan_buoy.py        已登记的站不重复登记；已存在的 csv 会被覆盖（内容只来自上述 nc）
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, 'metadata', 'sites.json')
RAW = os.path.join(ROOT, 'raw', 'insitu_taiwan_cmems')
OUT = os.path.join(ROOT, 'processed', 'buoy')
STRAIT = ['C6W08', '46757B', '46735A', '46787A', '46778A']
OTHER = ['46694A', '46708A', '46706A', '46699A', 'WRA007', '46714D', '46759A', 'C6V27', 'C6S62']
COLS = ['valid_time', 'hs', 'tp', 'tm02', 'mwd', 'wspd', 'wdir', 'pres', 'atmp', 'wtmp', 'hmax', 'label_tc_distance_km', 'event_id']


def convert(st):
    x = xr.open_dataset(os.path.join(RAW, f'GL_TS_MO_{st}.nc'))
    d = pd.DataFrame(dict(hs=x['VGHS'].values[:, 0].astype(float), qc=x['VGHS_QC'].values[:, 0].astype(float)), index=pd.to_datetime(x['TIME'].values))
    info = dict(alias=x.attrs['platform_name'], lat=round(float(x['LATITUDE'].values), 4), lon=round(float(x['LONGITUDE'].values), 4))
    x.close()
    d = d[(d.qc == 1) & d.hs.between(0, 25)]
    d = d[~d.index.duplicated()].sort_index()
    o = pd.DataFrame(index=d.index, columns=COLS[1:], dtype=float)
    o['hs'] = d.hs
    o.insert(0, 'valid_time', d.index.strftime('%Y-%m-%dT%H:%M:%SZ'))
    return o.reset_index(drop=True), info, d.index


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    text = open(META, encoding='utf-8').read()
    meta = json.loads(text)
    have = {s['id'] for s in meta['sites']}
    made = []
    for st in STRAIT + OTHER:
        sid = f'TW{st}'
        o, info, t = convert(st)
        o.to_csv(os.path.join(OUT, f'{sid}_{st}.csv'), index=False)
        made.append(sid)
        print(f"{sid} {info['alias']}：{len(o):,} 行，{t.min():%Y-%m-%d} 至 {t.max():%Y-%m-%d}，平均 {o.hs.mean():.2f} m")
        if sid not in have:
            meta['sites'].append(dict(
                id=sid, name=f"taiwan_{info['alias'].lower().replace(' ', '_').replace('-', '_')}",
                label=f"台湾中央气象署浮标 {info['alias']}（11-02 新增，{'台湾海峡' if st in STRAIT else '台湾周边'}）",
                point=[info['lat'], info['lon']], priority=9,
                buoy=dict(source='CMEMS', station=st, alias=info['alias'], lat=info['lat'], lon=info['lon'], years=[int(t.min().year), int(t.max().year)],
                          note='Copernicus INSITU_GLO_WAV_DISCRETE_MY_013_045（202411，逐时，只有 Hs），2026-11-02 下载，见 prepare_cmems_taiwan_buoy.py', source_name='台湾中央气象署（经 Copernicus）'),
                verification=f'CWA {st}；' + ('不在 download_gefs_reforecast.py 的提取清单，服务器没有它的回算' if st == 'C6S62' else '在 download_gefs_reforecast.py 的提取清单')))
    if len(meta['sites']) > len(have):
        open(META, 'w', encoding='utf-8').write(json.dumps(meta, ensure_ascii=False, indent=1) + ('\n' if text.endswith('\n') else ''))
    # 台风标签（诊断用，不进模型）
    import label_typhoons as lt
    tr = lt.load_tracks()
    by_id = {s['id']: s for s in meta['sites']}
    os.makedirs(lt.OUT, exist_ok=True)
    for sid in made:
        lat, lon = by_id[sid]['point']
        lab = lt.nearest_for_site(tr, lat, lon)
        lab.to_csv(os.path.join(lt.OUT, f'tc_hourly_{sid}.csv'))
        st = sid[2:]
        df = lt.fill_labels(os.path.join(OUT, f'{sid}_{st}.csv'), lab)
        print(f"  {sid} 台风标签：{(df['event_id'] != '').sum()} 行属于气旋事件")


if __name__ == '__main__':
    main()
