"""GEFS 海浪回算 + 浮标 → wavecast 训练数据集（真实预报试验 S5）。

输入：processed/gefs_reforecast/<站号>/<年>.csv.gz   每周三 11 个成员、逐时、35 天（hs、tr=Te、fp、th1m）
      processed/buoy/<站点ID>_<站号>.csv               浮标真值（含台风标签）
      processed/site_<ID>/observations.csv              ERA5 逐 3 小时（作过去 7 天的 Te、波向、风、气压、气温）
输出：processed/site_<ID>_reforecast/ 下的 observations.csv、forecasts.csv、manifest.json（mode = reforecast）
做法：
  - forecasts.csv：每次起报取 +3 h 至 +16 d 的逐 3 小时时刻，hs = 集合平均、hs_spread = 集合标准差、
    te = tr 集合平均（10-16 已核实 tr 即能量周期 Te）、mwd = th1m 的圆平均、q000–q098 = hs 集合分位数
    （概率水平 (i+0.5)/99，供"原始集合"基线评分）、n_members；
    available_time = 起报 + 6 小时（GEFS 00 时起报的业务产品一般在 4–6 小时后可得；回算本身是事后生成的，
    这是按业务情形设定的假设，写入 manifest）。
  - observations.csv：hs = 浮标（整点 ±30 分钟平均，≤ 6 小时的缺测线性插补）；te、mwd、u10、v10、t2m、msl 取 ERA5；
    台风标签取浮标文件（只作诊断）。available_time = valid_time + 1 小时。
用法：python prepare_reforecast_dataset.py [G K H]
10-23 加（接入服务器提取的平日回算，样本约 5 倍）：
  --sources 目录 …   processed/ 下的回算目录，可多个，合并后按（起报, 成员, 时刻）去重（服务器中断时可能写了重复行）；
                     默认只用 gefs_reforecast（周三）。平日：gefs_reforecast_weekday（及补提的 gefs_reforecast_weekday_early、
                     周三新站 gefs_reforecast_v2）
  --members5         所有起报都只用 c00、p01–p04 五个成员（平日只有 5 个；周三也取同样 5 个，使集合离散度、分位数等
                     输入在不同起报之间含义一致）
  --suffix _daily    输出到 processed/site_<ID>_reforecast<后缀>（不覆盖原来的周三数据集）
  --out-root 目录    输出根目录（默认 processed，测试时可指向临时目录）
  例：python prepare_reforecast_dataset.py G K H --sources gefs_reforecast gefs_reforecast_weekday --members5 --suffix _daily
10-27 加（独立业务检验，download_gefs_operational.py 的输出）：
  python prepare_reforecast_dataset.py G K H --sources gefs_operational --suffix _oper
11-04 加 --truth-prefix CW（台湾海峡独立检验）：站点 ID 不变（冻结模型的站点编号要对得上，如 TW46757B），真值改读
  processed/buoy/<前缀><站号>_<站号>.csv（气象署历史资料，2020 年后仍有观测；Copernicus 版止于 2017-03）。写入 manifest。
  例：python prepare_reforecast_dataset.py TW46757B TWC6W08 --sources gefs_operational_tws --suffix _oper --truth-prefix CW
  业务 GRIB 没有能量周期 Te：下载时存 MWSPER（≈ Tm02）为 tm02 列、tr 列为空。这里检测到"tr 全空、有 tm02"就按站换算：
  te 系数 = 回算 tr/ERA5 Te ÷ 业务 MWSPER/ERA5 Te（各取 c00、时效 6–24 h 的中位数；回算比值见
  metadata/gefs_reforecast_era5_ratios.json，业务比值用下载的全部数据现算），系数写进 manifest。只用 ERA5，不用浮标实测。
  --hs-era5-bridge   敏感性试验：波高也按同样方式对齐到回算（10-27 抽样：夏威夷业务 Hs 相对 ERA5 比回算高约 7%，
                     关岛 −2%、马罗岛 +4%）。主结果不加（业务上就是直接用 GEFS 原始波高）；建议加 --suffix _oper_hsbridge。
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = {s['id']: s for s in json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']}
TAU = (np.arange(99) + .5) / 99
LEAD_HOURS = np.arange(3, 16 * 24 + 1, 3)
AVAIL_DELAY_H = 6
GAP_FILL_H = 6


MEMBERS5 = ['c00', 'p01', 'p02', 'p03', 'p04']
COLS = ['issue_time', 'member', 'valid_time', 'hs', 'tr', 'th1m']
OPER_COLS = COLS + ['tm02']                      # 业务预报多一列 MWSPER（见 harmonize_operational）
RATIOS = os.path.join(ROOT, 'metadata', 'gefs_reforecast_era5_ratios.json')


def read_gz_tolerant(path):
    """Read a .csv.gz written by many appends (one gzip member per append), skipping damaged members.

    A download process killed mid-write leaves a truncated member; later appends follow it, so a normal read stops
    there. Here each member is decompressed on its own; a broken one is skipped by searching for the next gzip
    header. Rows lost this way belong to (date, member) jobs never marked done, so a resumed download re-writes them.
    Returns (DataFrame with COLS, number of skipped members)."""
    import io
    import zlib
    data = open(path, 'rb').read()
    parts, pos, bad = [], 0, 0
    while pos < len(data):
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            chunk = d.decompress(data[pos:])
            if not d.eof:                                   # truncated tail (file still being written, or killed)
                bad += 1
                break
            parts.append(chunk)
            pos = len(data) - len(d.unused_data)
        except zlib.error:
            bad += 1
            nxt = data.find(b'\x1f\x8b\x08', pos + 1)
            if nxt < 0:
                break
            pos = nxt
    lines = b''.join(parts).decode('utf-8', 'replace').splitlines()
    header = next((ln for ln in lines if ln.startswith('valid_time')), None)
    if header is None:
        return pd.DataFrame(columns=COLS), bad
    body = '\n'.join(ln for ln in lines if ln and not ln.startswith('valid_time'))
    df = pd.read_csv(io.StringIO(header + '\n' + body), usecols=lambda c: c in OPER_COLS, on_bad_lines='skip')
    return df, bad


def load_reforecast(station, sources=('gefs_reforecast',), members5=False):
    parts = []
    for src in sources:
        files = sorted(glob.glob(os.path.join(ROOT, 'processed', src, station, '*.csv.gz')))
        if not files:      # 10-24：服务器上没有本机的 gefs_reforecast（周三），曾被悄悄跳过，导致数据集缺周三
            print(f'    [警告] processed/{src}/{station} 下没有回算文件，这个来源被跳过', flush=True)
        for f in files:
            try:
                d = pd.read_csv(f, usecols=lambda c: c in OPER_COLS)
            except Exception as e:                           # damaged gzip member (killed / still writing)
                d, bad = read_gz_tolerant(f)
                print(f'    [注意] {os.path.relpath(f, ROOT)}：{str(e)[:60]}；跳过 {bad} 段损坏数据，读回 {len(d)} 行', flush=True)
            d = d.dropna(subset=['issue_time', 'valid_time'])
            if members5:
                d = d[d['member'].isin(MEMBERS5)]
            d['valid_time'] = pd.to_datetime(d['valid_time'], format='ISO8601').dt.round('1min')
            d['issue_time'] = pd.to_datetime(d['issue_time'], format='ISO8601')
            d['lead_h'] = ((d['valid_time'] - d['issue_time']) / pd.Timedelta('1h')).round().astype(int)
            parts.append(d[d['lead_h'].isin(LEAD_HOURS)])          # 先按时效筛掉逐时行，省内存
    if not parts:
        raise SystemExit(f'{station}：{sources} 里没有回算文件')
    df = pd.concat(parts, ignore_index=True)
    n = len(df)
    df = df.drop_duplicates(['issue_time', 'member', 'valid_time'])
    if len(df) < n:
        print(f'    {station}：去掉重复行 {n - len(df)} 行', flush=True)
    return df


def harmonize_operational(sid, rf, hs_bridge=False):
    """业务预报 → 与回算同义的输入，以本站 ERA5 格点为桥梁（不用浮标实测）。返回 (rf, 换算记录)。"""
    ref = json.load(open(RATIOS, encoding='utf-8'))['sites'][sid]
    e = pd.read_csv(os.path.join(ROOT, 'processed', f'site_{sid}', 'observations.csv'), usecols=['valid_time', 'hs', 'te'])
    e = e.drop_duplicates('valid_time')
    e.index = pd.to_datetime(e['valid_time'].str.replace('Z', ''))
    c = rf[(rf['member'] == 'c00') & rf['lead_h'].isin([6, 12, 18, 24])]
    c = c.join(e[['hs', 'te']].rename(columns={'hs': 'era5_hs', 'te': 'era5_te'}), on='valid_time', how='inner').dropna(
        subset=['hs', 'tm02', 'era5_hs', 'era5_te'])
    if len(c) < 50:
        raise SystemExit(f'{sid}：业务预报与 ERA5 重叠的 c00 样本只有 {len(c)} 个，无法换算周期')
    rec = dict(n_calibration=len(c), reforecast_tr_over_era5_te=ref['tr_over_era5_te'],
               operational_mwsper_over_era5_te=float(np.median(c['tm02'] / c['era5_te'])))
    rec['te_scale'] = rec['reforecast_tr_over_era5_te'] / rec['operational_mwsper_over_era5_te']
    rf = rf.copy()
    rf['tr'] = rf['tm02'] * rec['te_scale']
    rec['operational_hs_over_era5_hs'] = float(np.median(c['hs'] / c['era5_hs']))
    rec['reforecast_hs_over_era5_hs'] = ref['hs_over_era5_hs']
    rec['hs_scale'] = rec['reforecast_hs_over_era5_hs'] / rec['operational_hs_over_era5_hs'] if hs_bridge else 1.0
    rf['hs'] = rf['hs'] * rec['hs_scale']
    print(f"    {sid}：业务 MWSPER → Te 系数 {rec['te_scale']:.3f}；波高系数 {rec['hs_scale']:.3f}"
          f"（业务 / 回算 Hs 相对 ERA5：{rec['operational_hs_over_era5_hs']:.3f} / {rec['reforecast_hs_over_era5_hs']:.3f}）", flush=True)
    return rf, rec


def forecasts(site_id, rf):
    rows = []
    for issue, g in rf.groupby('issue_time'):
        hs = g.pivot(index='valid_time', columns='member', values='hs')
        if hs.shape[1] < 5 or len(hs) < len(LEAD_HOURS):
            continue
        tr = g.pivot(index='valid_time', columns='member', values='tr').reindex(hs.index)
        th = np.deg2rad(g.pivot(index='valid_time', columns='member', values='th1m').reindex(hs.index))
        x = hs.to_numpy()
        q = np.nanquantile(x, TAU, axis=1).T
        d = pd.DataFrame({'site_id': site_id, 'issue_time': issue, 'valid_time': hs.index,
                          'available_time': issue + pd.Timedelta(hours=AVAIL_DELAY_H),
                          'hs': np.nanmean(x, 1), 'hs_spread': np.nanstd(x, 1, ddof=1), 'te': tr.mean(axis=1).to_numpy(),
                          'mwd': (np.rad2deg(np.arctan2(np.nansum(np.sin(th), 1), np.nansum(np.cos(th), 1))) + 360) % 360,
                          'n_members': np.isfinite(x).sum(1)})
        d = pd.concat([d.reset_index(drop=True), pd.DataFrame(q, columns=[f'q{k:03d}' for k in range(99)])], axis=1)
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    return out[np.isfinite(out['hs'])]


def observations(site, truth_prefix=None):
    bid = f"{truth_prefix}{site['buoy']['station']}" if truth_prefix else site['id']
    b = pd.read_csv(os.path.join(ROOT, 'processed', 'buoy', f"{bid}_{site['buoy']['station']}.csv"),
                    usecols=['valid_time', 'hs', 'label_tc_distance_km', 'event_id'], low_memory=False)
    b.index = pd.to_datetime(b['valid_time'].str.replace('Z', ''))
    hs = b['hs'].groupby(b.index.round('1h')).mean()
    hs = hs.reindex(pd.date_range(hs.index.min(), hs.index.max(), freq='1h'))
    hs = hs.interpolate(limit=GAP_FILL_H, limit_area='inside')
    lab = b[['label_tc_distance_km', 'event_id']].groupby(b.index.round('1h')).first()
    e = pd.read_csv(os.path.join(ROOT, 'processed', f"site_{site['id']}", 'observations.csv'),
                    usecols=['valid_time', 'te', 'mwd', 'u10', 'v10', 't2m', 'msl'])
    e.index = pd.to_datetime(e['valid_time'].str.replace('Z', ''))
    t = e.index[(e.index >= hs.index.min()) & (e.index <= hs.index.max())]
    o = pd.DataFrame({'site_id': site['id'], 'hs': hs.reindex(t).to_numpy()}, index=t)
    o = o.join(e[['te', 'mwd', 'u10', 'v10', 't2m', 'msl']]).join(lab)
    o.index.name = 'valid_time'
    o['available_time'] = (o.index + pd.Timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    o = o.reset_index()
    o['valid_time'] = o['valid_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return o


def run(sid, sources=('gefs_reforecast',), members5=False, suffix='', out_root=None, hs_bridge=False, truth_prefix=None):
    site = SITES[sid]
    st = site['buoy']['station']
    out = os.path.join(out_root or os.path.join(ROOT, 'processed'), f'site_{sid}_reforecast{suffix}')
    if os.path.exists(os.path.join(out, 'manifest.json')):
        print(f'[跳过] {out} 已存在（要重做请先移到回收站）')
        return
    os.makedirs(out, exist_ok=True)
    obs = observations(site, truth_prefix)
    rf = load_reforecast(st, sources, members5)
    oper = None
    if 'tm02' in rf and rf['tr'].isna().all() and rf['tm02'].notna().any():
        rf, oper = harmonize_operational(sid, rf, hs_bridge)
    fc = forecasts(sid, rf)
    t0, t1 = pd.Timestamp(obs['valid_time'].iloc[0].replace('Z', '')), pd.Timestamp(obs['valid_time'].iloc[-1].replace('Z', ''))
    fc = fc[(fc['issue_time'] >= t0 - pd.Timedelta(days=7)) & (fc['issue_time'] <= t1)]
    for c in ('issue_time', 'valid_time', 'available_time'):
        fc[c] = pd.to_datetime(fc[c]).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    obs.to_csv(os.path.join(out, 'observations.csv'), index=False)
    fc.round(4).to_csv(os.path.join(out, 'forecasts.csv'), index=False)
    manifest = dict(mode='reforecast', site_id=sid, availability_verified=False,
                    forecast_source=f'NOAA GEFSv12 wave reforecast, station {st}, sources {list(sources)}, '
                                    + ('members c00,p01-p04 only' if members5 else 'all members available'),
                    truth_source=(f"CWA history buoy {truth_prefix}{st} (Ocean Weather Information Platform)" if truth_prefix else f"{site['buoy']['source']} buoy {st}")
                                 + f" (hourly mean of ±30 min, gaps ≤ {GAP_FILL_H} h interpolated)",
                    past_context='te, mwd, u10, v10, t2m, msl from CDS ERA5 at site grid point',
                    available_time_assumption=f'forecast available issue + {AVAIL_DELAY_H} h (operational GEFS 00Z typical); '
                                              'observation available valid_time + 1 h',
                    period_definition='te = GEFS tr ensemble mean, verified ≈ energy period Te (ratio to ERA5 mwp 1.02)',
                    n_issues=int(fc['issue_time'].nunique()), obs_period=[obs['valid_time'].iloc[0], obs['valid_time'].iloc[-1]])
    if oper:
        manifest.update(forecast_source=f'NOAA GEFSv12 OPERATIONAL wave ensemble (AWS noaa-gefs-pds, 0.25 deg grid interpolated to buoy {st}), '
                                        f'sources {list(sources)}; independent check, not used for any training or selection',
                        period_definition='te = MWSPER (approx. Tm02) x te_scale, te_scale from ERA5 bridge (see operational_harmonization)',
                        operational_harmonization=oper)
    json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    manifest['n_members_per_issue'] = {int(k): int(v) for k, v in fc.groupby('issue_time')['n_members'].max().value_counts().items()}
    json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print(f"[完成] {sid}：起报 {manifest['n_issues']} 次（每次成员数分布 {manifest['n_members_per_issue']}），预报 {len(fc)} 行，观测 {len(obs)} 行（Hs 有效 "
          f"{obs['hs'].notna().mean():.0%}），{manifest['obs_period'][0]} – {manifest['obs_period'][1]} → {out}")


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    args = sys.argv[1:]
    opts = {}
    for flag, many in (('--sources', True), ('--suffix', False), ('--out-root', False), ('--truth-prefix', False)):
        if flag in args:
            i = args.index(flag)
            vals = []
            for v in args[i + 1:]:
                if v.startswith('--'):
                    break
                vals.append(v)
                if not many:
                    break
            opts[flag] = vals
            del args[i:i + 1 + len(vals)]
    members5 = '--members5' in args
    hs_bridge = '--hs-era5-bridge' in args
    args = [a for a in args if a not in ('--members5', '--hs-era5-bridge')]
    for sid in (args or ['G', 'K', 'H']):
        run(sid, tuple(opts.get('--sources', ['gefs_reforecast'])), members5,
            (opts.get('--suffix') or [''])[0], (opts.get('--out-root') or [None])[0], hs_bridge, (opts.get('--truth-prefix') or [None])[0])
