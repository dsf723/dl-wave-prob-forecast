"""气象署海象环境资讯平台“资料浮标观测历史资料”（波浪）→ processed/buoy/CW<站号>_<站号>.csv，并在 metadata/sites.json 登记（11-04 事先登记的扩充训练 / 独立检验用）。

输入：raw/cwa_buoy_history/<站号>_<年>_wave.zip（每个一个 CSV：st,yyyymmddhhmi,H(cm),T=尖峰周期(0.1 s),Tmean=平均周期(0.1 s),D(度)；时间为 UTC，11-04 与 Copernicus 同站对过，RMSE 0.029 m）。
为什么用新 ID（CW 前缀）：冻结的 TWS 模型用的是 Copernicus 版 TW* 数据，不能改动；新竹、马祖与 TW46757B、TWC6W08 是同一个浮标。
质控：文件没有 QC 标记，只做合理性检查：0 < H ≤ 25 m，2 s ≤ 周期 ≤ 25 s，重复时刻取第一条。2006 年以前新竹每 2 小时一笔，原样保留。
列：与其他浮标文件相同；tp = 尖峰周期，tmean_cwa = 平均周期（定义未写明，暂不放进 tm02），mwd = 波向。
ERA5：新竹、马祖沿用 TW 站已下载的 ERA5 序列（name 相同；ERA5 波浪格点相同，浮标坐标差 0.1 km、3.6 km）；七美、东吉岛、台中暂不下载 ERA5。
用法：python prepare_cwa_buoy.py        已登记的站不重复登记；csv 覆盖重写（内容只来自上述 zip）
"""
import glob
import io
import json
import os
import sys
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, 'metadata', 'sites.json')
RAW = os.path.join(ROOT, 'raw', 'cwa_buoy_history')
OUT = os.path.join(ROOT, 'processed', 'buoy')
# 站号: (名称, 纬度, 经度, 水深与位置说明, 用途)，坐标与说明取自下载页面
STATIONS = {'46757B': ('taiwan_hsinchu', 24.7625, 120.84444, '海山漁港外海，離岸約6.4公里，水深約24公尺', '扩充训练与独立检验'),
            'C6W08': ('taiwan_matsu', 26.3525, 120.50861, '東引島東南方約3公里，水深約51公尺', '扩充训练与独立检验'),
            'C6W10': ('taiwan_qimei', 23.19444, 119.66472, '七美嶼東方23公里，水深50公尺', '只作 2020 年后参考'),
            'C5W09': ('taiwan_dongji', 23.2565, 119.6839, '東吉島東方，水深20公尺', '只作参考'),
            'C6F01': ('taiwan_taichung', 24.2397, 120.4199, '大肚溪出海口西北約6公里，水深19.8公尺', '只作 2020 年后参考')}
COLS = ['valid_time', 'hs', 'tp', 'tm02', 'tmean_cwa', 'mwd', 'wspd', 'wdir', 'pres', 'atmp', 'wtmp', 'hmax', 'label_tc_distance_km', 'event_id']


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode('utf-8-sig')
    lines = text.splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith('st,'))
    d = pd.read_csv(io.StringIO('\n'.join(lines[k:])), dtype={'yyyymmddhhmi': str})
    d.index = pd.to_datetime(d.pop('yyyymmddhhmi'), format='%Y%m%d%H%M')
    return d.apply(pd.to_numeric, errors='coerce')


def convert(st):
    d = pd.concat([read_zip(p) for p in sorted(glob.glob(os.path.join(RAW, f'{st}_*_wave.zip')))])
    d = d[~d.index.duplicated()].sort_index()
    hs = d.H / 100
    hs = hs.where((hs > 0) & (hs <= 25))
    per = lambda x: (x / 10).where((x / 10).between(2, 25))
    o = pd.DataFrame(index=d.index, columns=COLS[1:], dtype=float)
    o['hs'], o['tp'], o['tmean_cwa'], o['mwd'] = hs, per(d['T']), per(d.Tmean), d.D.where(d.D.between(0, 360))
    o['tm02'] = float('nan')
    o = o[o.hs.notna()]
    t = o.index
    o.insert(0, 'valid_time', t.strftime('%Y-%m-%dT%H:%M:%SZ'))
    return o.reset_index(drop=True), t


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    text = open(META, encoding='utf-8').read()
    meta = json.loads(text)
    have = {s['id'] for s in meta['sites']}
    n0, made = len(meta['sites']), []
    for st, (name, lat, lon, where, use) in STATIONS.items():
        sid = f'CW{st}'
        o, t = convert(st)
        o.to_csv(os.path.join(OUT, f'{sid}_{st}.csv'), index=False)
        made.append(sid)
        print(f'{sid} {name}：{len(o):,} 行，{t.min():%Y-%m-%d} 至 {t.max():%Y-%m-%d}，平均 Hs {o.hs.mean():.2f} m，Tp 有值 {o.tp.notna().mean():.0%}，Tmean 有值 {o.tmean_cwa.notna().mean():.0%}')
        if sid not in have:
            meta['sites'].append(dict(
                id=sid, name=name, label=f'台湾气象署浮标 {name.split("_")[1]} {st}（11-04 新增，气象署历史资料，{use}）',
                point=[lat, lon], priority=10,
                buoy=dict(source='CWA', station=st, lat=lat, lon=lon, years=[int(t.min().year), int(t.max().year)],
                          note=f'{where}；海象环境资讯平台 资料浮标观测历史资料（Wave-his），2026-11-04 下载，见 prepare_cwa_buoy.py', source_name='台湾中央气象署'),
                verification='与 Copernicus 同站逐时对比一致（新竹 2015 年 RMSE 0.029 m）' if st == '46757B' else '未与其他来源对比'))
    if len(meta['sites']) > n0:
        open(META, 'w', encoding='utf-8').write(json.dumps(meta, ensure_ascii=False, indent=1) + ('\n' if text.endswith('\n') else ''))
    import label_typhoons as lt
    tr = lt.load_tracks()
    by_id = {s['id']: s for s in meta['sites']}
    for sid in made:
        lat, lon = by_id[sid]['point']
        lab = lt.nearest_for_site(tr, lat, lon)
        lab.to_csv(os.path.join(lt.OUT, f'tc_hourly_{sid}.csv'))
        df = lt.fill_labels(os.path.join(OUT, f'{sid}_{sid[2:]}.csv'), lab)
        print(f'  {sid} 台风标签：{(df["event_id"] != "").sum()} 行属于气旋事件')


if __name__ == '__main__':
    main()
