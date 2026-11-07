"""ERA5 与浮标实测的独立对比（论文"独立验证"部分的第一步）。

真值：processed/buoy/<ID>_<站号>.csv（NDBC，30 分钟一条 → 取整点前后 30 分钟平均成逐时）
ERA5：优先 processed/site_<ID>/observations_hourly.csv（CDS 下载、era5_to_observations.py 生成）；
      没有时用 Open-Meteo 的 ERA5 海浪（era5_ocean，同一套再分析，经 Open-Meteo 提供）做"预览"，结果文件名带 _preview。
比较量：有效波高 Hs（浮标 WVHT vs ERA5 swh）。周期定义不同（浮标 DPD/APD vs ERA5 mwp），此处不直接比较。
输出：results/era5_vs_buoy/<ID>_metrics[_preview].csv（总体、按浪高、按季节、按台风距离分组）与 <ID>_hs[_preview].png
用法：python compare_era5_buoy.py [G H]
"""
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
OUT = os.path.join(ROOT, 'results', 'era5_vs_buoy')


def buoy_hourly(site):
    b = site['buoy']
    df = pd.read_csv(os.path.join(ROOT, 'processed', 'buoy', f"{site['id']}_{b['station']}.csv"), low_memory=False)
    t = pd.to_datetime(df['valid_time'].str.replace('Z', ''))
    s = pd.Series(df['hs'].values, index=t).dropna()
    hourly = s.groupby(s.index.round('1h')).mean()          # 整点 ±30 分钟内的平均
    lab = df[['label_tc_distance_km']].set_index(t)['label_tc_distance_km'] if 'label_tc_distance_km' in df else None
    lab = lab.groupby(lab.index.round('1h')).min() if lab is not None else None
    return hourly, lab


def era5_hs(site, start, end):
    p = os.path.join(ROOT, 'processed', f"site_{site['id']}", 'observations_hourly.csv')
    if os.path.exists(p):
        df = pd.read_csv(p, usecols=['valid_time', 'hs'])
        return pd.Series(df['hs'].values, index=pd.to_datetime(df['valid_time'].str.replace('Z', ''))), 'CDS ERA5', '', None
    lat, lon = site['buoy']['lat'], site['buoy']['lon']
    u = (f'https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&start_date={start:%Y-%m-%d}'
         f'&end_date={end:%Y-%m-%d}&hourly=wave_height&models=era5_ocean&timezone=GMT&cell_selection=sea')
    d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'research'}), timeout=600).read())
    s = pd.Series(d['hourly']['wave_height'], index=pd.to_datetime(d['hourly']['time']), dtype=float)
    return s, 'Open-Meteo ERA5（预览）', '_preview', (d['latitude'], d['longitude'])


def stats(o, m):
    e = m - o
    return {'n': len(o), 'obs_mean': o.mean(), 'era5_mean': m.mean(), 'bias': e.mean(), 'rmse': np.sqrt((e ** 2).mean()),
            'si': np.sqrt(((e - e.mean()) ** 2).mean()) / o.mean(), 'corr': np.corrcoef(o, m)[0, 1],
            'obs_p99': o.quantile(0.99), 'era5_p99': m.quantile(0.99), 'obs_max': o.max(), 'era5_max': m.max()}


def run(site):
    obs, lab = buoy_hourly(site)
    era, src, tag, cell = era5_hs(site, obs.index.min(), min(obs.index.max(), pd.Timestamp('2024-12-31 23:00')))
    j = pd.concat({'obs': obs, 'era5': era}, axis=1).dropna()
    j = j[j.index <= '2024-12-31 23:00']
    if lab is not None:
        j['tc_km'] = lab.reindex(j.index)
    rows = [{'group': '总体', **stats(j['obs'], j['era5'])}]
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 99)]:
        k = j[(j['obs'] >= lo) & (j['obs'] < hi)]
        if len(k) > 30:
            rows.append({'group': f'实测 Hs {lo}–{hi if hi < 99 else "以上"} m', **stats(k['obs'], k['era5'])})
    for name, months in (('冬季 11–3 月', [11, 12, 1, 2, 3]), ('夏季 6–9 月', [6, 7, 8, 9]), ('转换期 4–5、10 月', [4, 5, 10])):
        k = j[j.index.month.isin(months)]
        rows.append({'group': name, **stats(k['obs'], k['era5'])})
    if 'tc_km' in j:
        for name, cond in (('台风 300 km 内', j['tc_km'] <= 300), ('台风 300–800 km', (j['tc_km'] > 300) & (j['tc_km'] <= 800)),
                           ('无台风（>800 km）', j['tc_km'] > 800)):
            k = j[cond]
            if len(k) > 30:
                rows.append({'group': name, **stats(k['obs'], k['era5'])})
    os.makedirs(OUT, exist_ok=True)
    res = pd.DataFrame(rows).round(3)
    res.insert(0, 'era5_source', src)
    res.to_csv(os.path.join(OUT, f"{site['id']}_metrics{tag}.csv"), index=False, encoding='utf-8-sig')
    print(f"\n[{site['id']}] {site['label']}  浮标 {site['buoy']['source']} {site['buoy']['station']} vs {src}"
          + (f"（ERA5 格点 {cell}）" if cell else '') + f"，{j.index.min():%Y-%m} – {j.index.max():%Y-%m}")
    print(res[['group', 'n', 'obs_mean', 'bias', 'rmse', 'si', 'corr', 'obs_p99', 'era5_p99', 'obs_max', 'era5_max']].to_string(index=False))
    plot(site, j, src, tag)
    if not tag:
        periods(site)


def periods(site):
    """周期对比（只用 CDS 数据）：ERA5 谱峰周期 tp（pp1d）vs 浮标 DPD，跨零周期 tm02（mp2）vs 浮标 APD；
    另给能量周期 te（mwp）vs 浮标 APD 作参考（定义不同，Te 应比 Tm02 长约 20%）。浮标没有 Te 的直接观测。"""
    p = os.path.join(ROOT, 'processed', f"site_{site['id']}", 'observations_hourly.csv')
    e = pd.read_csv(p, usecols=['valid_time', 'te', 'tp', 'tm02', 'label_tc_distance_km'])
    e.index = pd.to_datetime(e['valid_time'].str.replace('Z', ''))
    b = pd.read_csv(os.path.join(ROOT, 'processed', 'buoy', f"{site['id']}_{site['buoy']['station']}.csv"),
                    usecols=['valid_time', 'tp', 'tm02'])
    b.index = pd.to_datetime(b['valid_time'].str.replace('Z', ''))
    b = b[['tp', 'tm02']].groupby(b.index.round('1h')).mean()
    j = e[['te', 'tp', 'tm02', 'label_tc_distance_km']].join(b, rsuffix='_buoy')
    j = j[j.index <= '2024-12-31 23:00']
    rows = []
    for name, era, obs in (('谱峰周期 Tp：ERA5 pp1d vs 浮标 DPD', 'tp', 'tp_buoy'),
                           ('跨零周期 Tm02：ERA5 mp2 vs 浮标 APD', 'tm02', 'tm02_buoy'),
                           ('参考：ERA5 能量周期 Te vs 浮标 APD', 'te', 'tm02_buoy')):
        k = j[[era, obs, 'label_tc_distance_km']].dropna(subset=[era, obs])
        if len(k) < 100:
            continue
        groups = [('总体', k), ('冬季 11–3 月', k[k.index.month.isin([11, 12, 1, 2, 3])]),
                  ('夏季 6–9 月', k[k.index.month.isin([6, 7, 8, 9])]),
                  ('台风 300 km 内', k[k['label_tc_distance_km'] <= 300]), ('无台风（>800 km）', k[k['label_tc_distance_km'] > 800])]
        for g, kk in groups:
            if len(kk) > 30:
                rows.append({'variable': name, 'group': g, **stats(kk[obs], kk[era])})
    if not rows:
        return
    res = pd.DataFrame(rows).round(3)
    res.to_csv(os.path.join(OUT, f"{site['id']}_period_metrics.csv"), index=False, encoding='utf-8-sig')
    print(f"  周期对比（单位 s；obs = 浮标）：")
    print(res[['variable', 'group', 'n', 'obs_mean', 'bias', 'rmse', 'corr']].to_string(index=False))


def plot(site, j, src, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hexbin(j['obs'], j['era5'], gridsize=60, bins='log', cmap='Blues', mincnt=1)
    m = max(j['obs'].max(), j['era5'].max())
    ax[0].plot([0, m], [0, m], 'k--', lw=1)
    q = np.linspace(0.01, 0.999, 200)
    ax[0].plot(j['obs'].quantile(q), j['era5'].quantile(q), color='#D9822B', lw=2, label='分位数–分位数')
    ax[0].set_xlabel('浮标实测 Hs (m)'); ax[0].set_ylabel(f'{src} Hs (m)'); ax[0].legend(loc='upper left')
    ax[0].set_title(f"{site['id']} 散点（颜色为样本数，对数）")
    top = j['obs'].idxmax()
    w = j.loc[top - pd.Timedelta(days=5): top + pd.Timedelta(days=5)]
    ax[1].plot(w.index, w['obs'], color='#1B2A36', label='浮标实测')
    ax[1].plot(w.index, w['era5'], color='#1C6E7D', label='ERA5')
    ax[1].set_title(f'实测最大值前后 10 天（{top:%Y-%m-%d}）'); ax[1].set_ylabel('Hs (m)'); ax[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{site['id']}_hs{tag}.png"), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sel = set(sys.argv[1:])
    for s in SITES:
        b = s.get('buoy') or {}
        if b.get('source') in ('NDBC', 'KMA') and (not sel or s['id'] in sel) and \
                os.path.exists(os.path.join(ROOT, 'processed', 'buoy', f"{s['id']}_{b['station']}.csv")):
            run(s)
