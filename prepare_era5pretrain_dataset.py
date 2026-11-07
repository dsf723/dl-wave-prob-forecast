"""预训练数据集：浮标开始观测之前的年份，用 GEFS 回算作预报输入、ERA5 作临时真值。

为什么：真实回算 + 浮标的训练样本只有 730 个起报；留一站检验表明每个站都需要本地样本。浮标开始之前，
GEFS 回算（2000 年起）已经有了，但没有浮标真值——用 ERA5 在站点格点的 Hs 当临时真值做预训练，再用浮标微调。
年份（都早于各站的训练年份，与验证、测试年份不重叠）：关岛 2000–2003、马罗岛 2000–2007、夏威夷 2000–2011。
输出：processed/site_<ID>_era5pre/ 下的 observations.csv（ERA5）、forecasts.csv（回算）、manifest.json（mode = reforecast）
用法：python prepare_era5pretrain_dataset.py [G K H]
11-02 加（台湾海峡 5 站，浮标 2012-09 起，预训练 2000–2011；服务器上周三回算在 gefs_reforecast_v2）：
  python prepare_era5pretrain_dataset.py TWC6W08 TW46757B TW46735A TW46787A TW46778A --sources gefs_reforecast_v2
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import prepare_reforecast_dataset as prep        # noqa: E402

PRE_YEARS = {'G': (2000, 2003), 'K': (2000, 2007), 'H': (2000, 2011),
             'TWC6W08': (2000, 2011), 'TW46757B': (2000, 2011), 'TW46735A': (2000, 2011), 'TW46787A': (2000, 2011), 'TW46778A': (2000, 2011),
             'CWC6W08': (2000, 2009)}   # 11-04 扩充训练：马祖气象署记录 2010-08 起；新竹 1997 年起有浮标，不预训练


def run(sid, sources=('gefs_reforecast',)):
    site = prep.SITES[sid]
    y0, y1 = PRE_YEARS[sid]
    out = os.path.join(ROOT, 'processed', f'site_{sid}_era5pre')
    if os.path.exists(os.path.join(out, 'manifest.json')):
        print(f'[跳过] {out} 已存在')
        return
    os.makedirs(out, exist_ok=True)
    e = pd.read_csv(os.path.join(ROOT, 'processed', f'site_{sid}', 'observations.csv'))
    t = pd.to_datetime(e['valid_time'].str.replace('Z', ''))
    # keep 7 days before the first issue (past window) and 16 days after the last one (forecast window)
    keep = (t >= pd.Timestamp(f'{y0}-01-01') - pd.Timedelta(days=8)) & (t <= pd.Timestamp(f'{y1}-12-31') + pd.Timedelta(days=17))
    obs = e.loc[keep, ['site_id', 'valid_time', 'available_time', 'hs', 'te', 'mwd', 'u10', 'v10', 't2m', 'msl',
                       'label_tc_distance_km', 'event_id']].copy()
    obs['site_id'] = sid
    rf = prep.load_reforecast(site['buoy']['station'], sources)
    rf = rf[(rf['issue_time'] >= pd.Timestamp(f'{y0}-01-01')) & (rf['issue_time'] <= pd.Timestamp(f'{y1}-12-31'))]
    fc = prep.forecasts(sid, rf)
    for c in ('issue_time', 'valid_time', 'available_time'):
        fc[c] = pd.to_datetime(fc[c]).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    obs.to_csv(os.path.join(out, 'observations.csv'), index=False)
    fc.round(4).to_csv(os.path.join(out, 'forecasts.csv'), index=False)
    manifest = dict(mode='reforecast', site_id=sid, availability_verified=False, purpose='pretraining only',
                    forecast_source=f"NOAA GEFSv12 wave reforecast, station {site['buoy']['station']}, Wednesdays",
                    truth_source='CDS ERA5 Hs at the site grid point (proxy truth, not observations)',
                    years=[y0, y1], n_issues=int(fc['issue_time'].nunique()))
    json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[完成] {sid}：{y0}–{y1}，起报 {manifest['n_issues']} 次，ERA5 观测 {len(obs)} 行 → {out}")


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    args = sys.argv[1:]
    sources = ('gefs_reforecast',)
    if '--sources' in args:
        i = args.index('--sources')
        sources, args = tuple(args[i + 1:]), args[:i]
    for sid in (args or ['G', 'K', 'H']):
        run(sid, sources)
