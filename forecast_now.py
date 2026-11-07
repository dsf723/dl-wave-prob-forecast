"""单次实时起报一键运行（10-28）：取一次 GEFS 00 时海浪集合起报，用冻结的最终模型出 30 天概率预报表。

    python forecast_now.py                              # 最新一次完整的 00 时起报，三站 G K H
    python forecast_now.py --date 20261028              # 指定起报日
    python forecast_now.py --obs K=recent_K.csv         # 某站近 7 天浮标实测（UTC，列 valid_time, hs），优先于其他来源
    python forecast_now.py --date 20211208 --check      # 历史起报核对：输入与业务独立检验完全相同（本机下载过的业务预报、
                                                        #   浮标、ERA5），预测应与 results/operational/<A>_ens_oper/quantiles.npz 一致
流程（与业务独立检验同一条代码路径，不训练、不挑选、不调参）：
  1. 冻结核对：两个集成成员和周期模型的配置指纹与冻结时一致才运行（evaluate_operational.frozen）；
  2. 这次起报：download_gefs_operational.day_job，11 个成员 × f003–f384 × HTSGW / MWSPER / WWSDIR（约 1.9 GB；10-28 服务器经 AWS 24 分钟，默认改走 GCS 镜像），
     业务 MWSPER → Te 用业务检验时定下的系数（processed/site_<ID>_reforecast_oper/manifest.json 的 operational_harmonization，
     不重新估计；波高不换算，与主结果相同），整理方式同 prepare_reforecast_dataset.forecasts；
  3. 过去 7 天输入（决策时刻 = 起报 + 6 h，与训练相同）：
     - Hs：--obs 给的文件 → processed/buoy 的浮标资料（覆盖时）→ NDBC 实时文件（realtime2）→ 都没有时用 GEFS 近 7 次起报
       c00 的前 24 小时（f003–f024，除以业务 Hs / ERA5 比值换成 ERA5 口径）代替；浮标缺的个别时刻也用它补；
     - Te、波向：ERA5 站点序列（processed/site_<ID>/observations.csv，覆盖时）→ 否则同样取 GEFS 近期起报 c00 的前 24 小时
       （MWSPER ÷ 业务 MWSPER / ERA5 Te 比值）。ERA5 实时约滞后 5 天，实时运行一般走这一条。
     代价（10-28 验证集，冻结 Transformer）：过去 Hs 改用 ERA5，CRPS +0.9%（关岛 +2.2%，马罗岛、夏威夷约 0）；
     过去 Te / 波向整段缺失也只 +0.9%。10-28 查看时，NDBC 上 52200（关岛）、51207（夏威夷）近 45 天没有实时数据，
     马罗岛 22107（KMA）没有开放的实时接口 → 不给 --obs 时三站的过去 Hs 都由 GEFS 代替，结果里写明来源；
  4. 预测：Transformer、GRU 各 3 个种子平均，两者分位数平均（集成）；周期模型给 Te 分位数（有实测谱的关岛、夏威夷；
     马罗岛用 GEFS 原始 Te，与 export_quantile_forecasts 相同）；波能流由 Hs–Te 成对情景（高斯 copula）算；
  5. 第 16–30 天接气候态（训练年浮标、预见日期 ±15 天，export_quantile_forecasts.site_climatology）；模型最后 2 天按分位数
     逐步过渡到气候态（--blend-days，10-29 验证集选定，见 compare_blend_days.py；0 = 直接切换，与论文检验相同）。
输出：results/realtime/<起报日>/
  forecast_30d_<运行时刻>.csv   每站逐 3 小时：Hs 的 P05–P95、超过阈值的概率、GEFS 原始 P10/P50/P90、Te、波能流、来源
  summary_<运行时刻>.md         每站逐日一览、30 天平均波能与距平、输入来源与提醒
  run_<运行时刻>.json           输入来源、换算系数、冻结指纹、下载说明
  inputs/                      下载与整理好的输入（重跑同一起报日时复用，不再下载）
"""
import argparse
import concurrent.futures as cf
import copy
import datetime as dt
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
import download_gefs_operational as dl                                        # noqa: E402
import prepare_reforecast_dataset as prep                                     # noqa: E402
from evaluate_operational import frozen                                      # noqa: E402
from verify_release import predict_seeds                                     # noqa: E402
from compare_blend_days import blend_weight                                  # noqa: E402
from export_quantile_forecasts import (site_climatology, climatology_rows, add_period_energy, observed_te,  # noqa: E402
                                       LEVELS, FLUX)
import torch                                                                 # noqa: E402
from wavecast.data import feature_names                                      # noqa: E402
from wavecast.experiment import load_sites, build_all                        # noqa: E402
from wavecast.metrics import quantile_at, TAU                                # noqa: E402

RUNS = ['GKH_h15_transformer_pretrain', 'GKH_h15_gru_pretrain']
TE_RUN = 'GKH_h15_te_transformer_pretrain'
OUT = os.path.join(ROOT, 'results', 'realtime')
NDBC_RT = 'https://www.ndbc.noaa.gov/data/realtime2/{st}.txt'
MEMBER_VARS = ['HTSGW', 'MWSPER', 'WWSDIR']
RECENT_FH = list(range(3, 25, 3))                 # 近期起报只取前 24 小时，当作"过去"的近似分析场


def log(msg):
    print(f'{dt.datetime.now():%H:%M:%S} {msg}', flush=True)


def latest_issue(bases):
    """最近一次 c00、p10 的 f384 都已上传的 00 时起报日（GEFS 海浪 00 时起报一般 6–8 小时后传完）。"""
    ses = requests.Session()
    ses.headers['User-Agent'] = 'research'
    today = pd.Timestamp.now(tz='UTC').normalize()
    for k in range(4):
        day = (today - pd.Timedelta(days=k)).strftime('%Y%m%d')
        for base in bases:
            try:
                for m in ('c00', dl.MEMBERS[-1]):
                    dl.fetch(ses, dl.url_of(day, m, 384, base) + '.idx', tries=2)
                return day
            except dl.Missing:
                continue
    raise SystemExit('最近 4 天都没有完整的 GEFS 海浪 00 时起报（存档或网络问题）')


def harmonization(sid, override):
    """业务 → 回算口径的换算系数：业务检验时定下的值（不重新估计）。"""
    if override:
        rec = json.load(open(override, encoding='utf-8'))[sid]
    else:
        path = os.path.join(ROOT, 'processed', f'site_{sid}_reforecast_oper', 'manifest.json')
        if not os.path.exists(path):
            raise SystemExit(f'没有 {path}：换算系数来自业务检验数据集（prepare_reforecast_dataset.py --sources gefs_operational --suffix _oper）')
        rec = json.load(open(path, encoding='utf-8'))['operational_harmonization']
    return dict(te_scale=rec['te_scale'], hs_scale=1.0,
                mwsper_over_era5_te=rec['operational_mwsper_over_era5_te'], hs_over_era5_hs=rec['operational_hs_over_era5_hs'])


def tidy(df):
    """day_job / load_reforecast 的行 → 统一列（时间为不带时区的 UTC）。"""
    df = df.copy()
    df['valid_time'] = pd.to_datetime(df['valid_time'], format='ISO8601').dt.round('1min')
    df['issue_time'] = pd.to_datetime(df['issue_time'], format='ISO8601')
    df['lead_h'] = ((df['valid_time'] - df['issue_time']) / pd.Timedelta('1h')).round().astype(int)
    return df


def gefs_issue(day, stations, a, inputs, bases):
    """这次起报的 11 个成员；--check 时取本机业务检验下载过的行（与评分时的输入完全相同）。"""
    if a.check:
        parts = []
        for st in stations:
            d = prep.load_reforecast(st['station'], ('gefs_operational',))
            d = d[d['issue_time'] == pd.Timestamp(day)]
            parts.append(d.assign(station=st['station']))
        df = pd.concat(parts, ignore_index=True)
        if df.empty:
            raise SystemExit(f'processed/gefs_operational 里没有 {day} 的起报，无法核对')
        return df, '业务检验下载过的数据（processed/gefs_operational）'
    cache = os.path.join(inputs, f'gefs_{day}_members.csv.gz')
    if os.path.exists(cache):
        return tidy(pd.read_csv(cache, dtype={'station': str})), f'复用 {os.path.relpath(cache, ROOT)}'
    log(f'下载 {day} 00 时起报：{len(dl.MEMBERS)} 个成员 × {len(dl.FHOURS)} 个时效 × {MEMBER_VARS}（约 1.9 GB）…')
    df, nbytes, _, note = dl.day_job(day, dl.MEMBERS, stations, MEMBER_VARS, [], dl.FHOURS, a.threads, bases)
    if df is None:
        raise SystemExit(f'{day}：{note}')
    df.to_csv(cache, index=False, compression='gzip')
    log(f'下载完成 {nbytes / 1e9:.2f} GB，{len(df)} 行{("（" + note + "）") if note else ""}')
    return tidy(df), f'下载 {nbytes / 1e9:.2f} GB' + (f'；{note}' if note else '')


def gefs_recent(day, stations, a, inputs, bases, issue_rows):
    """近 7 次起报 c00 的 f003–f024（加上这次起报 c00 的 f003、f006）：每个有效时刻取时效最短的一条，作过去 7 天的近似分析场。"""
    cache = os.path.join(inputs, f'gefs_recent7_c00_{day}.csv.gz')
    if os.path.exists(cache):
        rec = tidy(pd.read_csv(cache, dtype={'station': str}))
    else:
        parts = []
        days = [(pd.Timestamp(day) - pd.Timedelta(days=k)).strftime('%Y%m%d') for k in range(1, 8)]
        # 10-28：服务器上逐日下载要 8 分钟（单连接慢、每天只有 8 条记录）→ 7 天同时下
        with cf.ThreadPoolExecutor(len(days)) as ex:
            jobs = [ex.submit(dl.day_job, d, ['c00'], stations, MEMBER_VARS, [], RECENT_FH, max(2, a.threads // 4), bases)
                    for d in days]
            for d, fu in zip(days, jobs):
                df, nbytes, _, note = fu.result()
                log(f'  近期起报 {d} c00 f003–f024：{"缺失（" + note + "）" if df is None else f"{nbytes / 1e6:.0f} MB"}')
                if df is not None:
                    parts.append(df)
        if not parts:
            return {}
        rec = pd.concat(parts, ignore_index=True)
        rec.to_csv(cache, index=False, compression='gzip')
        rec = tidy(rec)
    own = issue_rows[(issue_rows['member'] == 'c00') & (issue_rows['lead_h'] <= 6)]
    rec = pd.concat([rec, own], ignore_index=True).sort_values('lead_h')
    return {st: g.drop_duplicates('valid_time').set_index('valid_time').sort_index()[['hs', 'tm02', 'th1m']]
            for st, g in rec.groupby(rec['station'].astype(str))}


def hourly(times, hs):
    """浮标 → 整点（±30 分钟平均），≤ 6 小时的缺测线性插补（同 prepare_reforecast_dataset.observations）。"""
    s = pd.Series(np.asarray(hs, float), index=pd.DatetimeIndex(times)).dropna()
    s = s[s.index.notna()]
    if s.empty:
        return s
    s = s.groupby(s.index.round('1h')).mean()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='1h'))
    return s.interpolate(limit=prep.GAP_FILL_H, limit_area='inside')


def ndbc_realtime(station):
    r = requests.get(NDBC_RT.format(st=station), timeout=60, headers={'User-Agent': 'research'})
    if r.status_code != 200:
        return None, f'NDBC 实时文件 HTTP {r.status_code}'
    lines = r.text.splitlines()
    cols = lines[0].lstrip('#').split()
    d = pd.read_csv(io.StringIO('\n'.join(lines[2:])), sep=r'\s+', names=cols, na_values=['MM'])
    t = pd.to_datetime(dict(year=d['YY'], month=d['MM'], day=d['DD'], hour=d['hh'], minute=d['mm']), errors='coerce')
    return hourly(t, d['WVHT']), 'NDBC 实时（realtime2）'


def buoy_hs(sid, site, pi, user_files):
    """过去 7 天的浮标 Hs（逐 3 小时，可能有缺）和来源；按 --obs → 本机浮标资料 → NDBC 实时 的顺序取第一个有数据的。"""
    st = str(site['buoy']['station'])
    f = os.path.join(ROOT, 'processed', 'buoy', f'{sid}_{st}.csv')

    def user():
        d = pd.read_csv(a_obs[sid])
        return hourly(pd.to_datetime(d['valid_time'], utc=True).dt.tz_localize(None), d['hs']), f'--obs {a_obs[sid]}'

    def local():
        b = pd.read_csv(f, usecols=['valid_time', 'hs'], low_memory=False)
        return hourly(pd.to_datetime(b['valid_time'].str.replace('Z', '')), b['hs']), f'本机浮标资料 {os.path.relpath(f, ROOT)}'

    def ndbc():
        try:
            s, src = ndbc_realtime(st)
        except requests.RequestException as e:
            s, src = None, f'NDBC 实时文件读取失败（{str(e)[:60]}）'
        return (s if s is not None else pd.Series(dtype=float)), src

    a_obs = user_files
    tries = ([user] if sid in user_files else []) + ([local] if os.path.exists(f) else []) + \
            ([ndbc] if site['buoy'].get('source') == 'NDBC' else [])
    notes = []
    for get in tries:
        s, src = get()
        v = s.reindex(pi) if len(s) else pd.Series(np.nan, index=pi)
        if v.notna().any():
            return v, src, notes
        notes.append(f'{src}：这 7 天没有数据')
    if site['buoy'].get('source') == 'KMA' and sid not in user_files:
        notes.append('KMA 浮标没有开放的实时接口：有近 7 天实测时用 --obs 提供')
    return pd.Series(np.nan, index=pi), None, notes


def era5_context(sid, pi):
    f = os.path.join(ROOT, 'processed', f'site_{sid}', 'observations.csv')
    if not os.path.exists(f):
        return None
    e = pd.read_csv(f, usecols=['valid_time', 'te', 'mwd'])
    e.index = pd.to_datetime(e['valid_time'].str.replace('Z', ''))
    e = e[~e.index.duplicated()].reindex(pi)
    return e if e[['te', 'mwd']].notna().all().all() else None


def write_site(sid, site, rf, h, pi, buoy, era5, recent, a, inputs):
    """一个站的实时数据集（与 prepare_reforecast_dataset.py 的输出同一格式）：过去 7 天的观测 + 这次起报。返回来源说明。"""
    st = str(site['buoy']['station'])
    info = dict(station=st, notes=[])
    hs, src, notes = buoy
    info['notes'] += notes
    prox = recent.get(st)
    prox = prox.reindex(pi) if prox is not None else pd.DataFrame(index=pi, columns=['hs', 'tm02', 'th1m'], dtype=float)
    hs_proxy = prox['hs'] / h['hs_over_era5_hs']
    fill = hs.isna() & hs_proxy.notna()
    if src is None and a.require_buoy:
        raise SystemExit(f'{sid}：过去 7 天没有浮标实测（{"；".join(notes)}），--require-buoy 不允许用 GEFS 代替')
    hs = hs.where(~fill, hs_proxy)
    info['past_hs'] = (src or 'GEFS 近期起报 c00 前 24 小时（ERA5 口径）') + (f'；{int(fill.sum())} 个时刻用 GEFS 补' if src and fill.any() else '')
    info['past_hs_from_gefs'] = int(fill.sum())
    if era5 is not None:
        te, mwd, info['past_te_mwd'] = era5['te'], era5['mwd'], 'ERA5 站点序列'
    else:
        te, mwd = prox['tm02'] / h['mwsper_over_era5_te'], prox['th1m']
        info['past_te_mwd'] = 'GEFS 近期起报 c00 前 24 小时（Te 换成 ERA5 口径）'
        if te.isna().any():
            info['notes'].append(f'过去 Te / 波向有 {int(te.isna().sum())} 个时刻缺（模型按缺测处理，验证集上整段缺失的代价约 +0.9%）')
    if hs.isna().any():
        info['skip'] = f'过去 7 天 Hs 有 {int(hs.isna().sum())} 个时刻既无浮标也无 GEFS 近期起报，不能预报'
        return info
    d = os.path.join(inputs, f'site_{sid}')
    os.makedirs(d, exist_ok=True)
    obs = pd.DataFrame({'site_id': sid, 'hs': hs.to_numpy(), 'te': te.to_numpy(), 'mwd': mwd.to_numpy(),
                        'label_tc_distance_km': np.nan, 'event_id': np.nan}, index=pi)
    obs.index.name = 'valid_time'
    obs['available_time'] = (obs.index + pd.Timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    obs = obs.reset_index()
    obs['valid_time'] = obs['valid_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    g = rf[(rf['station'].astype(str) == st) & rf['lead_h'].isin(prep.LEAD_HOURS)].copy()
    g['tr'], g['hs'] = g['tm02'] * h['te_scale'], g['hs'] * h['hs_scale']
    fc = prep.forecasts(sid, g)
    if fc.empty:
        info['skip'] = 'GEFS 这次起报的成员或时效不全（少于 5 个成员或缺时效），不能预报'
        return info
    for c in ('issue_time', 'valid_time', 'available_time'):
        fc[c] = pd.to_datetime(fc[c]).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    obs.to_csv(os.path.join(d, 'observations.csv'), index=False)
    fc.round(4).to_csv(os.path.join(d, 'forecasts.csv'), index=False)
    json.dump(dict(mode='reforecast', site_id=sid, availability_verified=False,
                   forecast_source=f'NOAA GEFSv12 operational wave ensemble, buoy point {st}, issue {fc["issue_time"].iloc[0]}',
                   truth_source='none (real-time run; no future observations)',
                   past_context=f"hs: {info['past_hs']}; te, mwd: {info['past_te_mwd']}",
                   period_definition='te = MWSPER x te_scale (frozen from the operational check)', harmonization=h,
                   n_issues=1, obs_period=[obs['valid_time'].iloc[0], obs['valid_time'].iloc[-1]]),
              open(os.path.join(d, 'manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    info['members'] = int(fc['n_members'].max())
    return info


def realtime_cfg(cfg, sids, inputs, years):
    new = copy.deepcopy(cfg)
    new['sites'] = []
    for s in cfg['sites']:
        if s['site_id'] in sids:
            if set(years) & (set(s['train_years']) | set(s['val_years'])):
                raise SystemExit('起报年份与训练 / 验证年份重叠：这一天属于训练或挑选用过的数据')
            new['sites'].append(dict(s, data_dir=os.path.join(inputs, f"site_{s['site_id']}").replace('\\', '/'), test_years=list(years)))
    return new


def samples_for(cfg, ablation, sids, inputs, years, decision):
    """与 build_samples 同一条路径；只是这次没有未来实测：未来时刻放占位值（只进 y，不进任何输入，输出时丢掉）。"""
    rt = realtime_cfg(cfg, sids, inputs, years)
    sites, _ = load_sites(rt)
    step, horizon = cfg['step_hours'], cfg['horizon_days'] * 24 // cfg['step_hours']
    fi = pd.date_range(start=decision + pd.Timedelta(hours=step), periods=horizon, freq=f'{step}h', tz='UTC')
    with_future = []
    for c, obs, fc in sites:
        fut = pd.DataFrame({'site_id': obs['site_id'].iloc[0], 'hs': 0.0, 'available_time': fi}, index=fi)
        fut.index.name = obs.index.name
        with_future.append((c, pd.concat([obs[obs.index <= pd.Timestamp(decision, tz='UTC')], fut]), fc))
    sites = with_future
    layout = [str(s['site_id']) for s in cfg['sites']]
    smp, skipped, names = build_all(sites, rt, feature_names(cfg, ablation), site_ids=layout)
    return smp['test'], skipped, names, sites


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--date', help='起报日 YYYYMMDD（默认最近一次完整的 00 时起报）')
    p.add_argument('--sites', nargs='+', default=['G', 'K', 'H'])
    p.add_argument('--obs', nargs='*', default=[], help='SITE=文件.csv：该站近 7 天浮标实测（列 valid_time（UTC）, hs）')
    p.add_argument('--require-buoy', action='store_true', help='没有浮标实测的站不预报（默认用 GEFS 近期起报代替，结果里注明）')
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--blend-days', type=int, default=2, help='模型最后几天逐步过渡到气候态（10-29 验证集选定 2；0 = 直接切换，与论文检验相同）')
    p.add_argument('--threshold', type=float, nargs='+', default=[2.5], help='输出超过这些波高（m）的概率')
    p.add_argument('--no-te', action='store_true', help='不跑周期模型（Te 用 GEFS 原始值）')
    p.add_argument('--copula-table', default=os.path.join(ROOT, 'results', 'joint', 'copula_rho_pit.csv'))
    p.add_argument('--harmonization', help='换算系数 JSON（{站: operational_harmonization}），默认读 _oper 数据集的 manifest')
    p.add_argument('--mirror', choices=list(dl.MIRRORS), default='gcs',
                   help='默认 gcs（10-27 服务器实测比 AWS 快），镜像没有的文件自动退回 AWS')
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--check', action='store_true', help='历史起报核对（见文件开头）')
    p.add_argument('--runs', nargs='+', default=RUNS, help='测试用：只用部分集成成员')
    a = p.parse_args()
    a.obs = dict(x.split('=', 1) for x in a.obs)
    t_run = pd.Timestamp.now(tz='UTC')
    stamp = t_run.strftime('%Y%m%d%H%M')
    bases = (dl.MIRRORS[a.mirror],) + ((dl.BUCKET,) if a.mirror != 'aws' else ())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. frozen models
    runs = {r: frozen(r) for r in a.runs}
    te_run = None if a.no_te else frozen(TE_RUN)
    cfg_a, _, abl_a = runs[a.runs[0]]
    if any(feature_names(c, ab) != feature_names(cfg_a, abl_a) for c, _, ab in runs.values()):
        raise SystemExit('集成成员的输入特征不同，样本无法对齐')
    log(f'冻结核对通过：{", ".join(f"{r}（{m}_{ab}）" for r, (_, m, ab) in runs.items())}' + ('' if a.no_te else f'；周期模型 {TE_RUN}'))

    # 2. this issue
    day = a.date or latest_issue(bases)
    issue = pd.Timestamp(day)
    decision = issue + pd.Timedelta(hours=cfg_a['decision_delay_hours'])
    step = cfg_a['step_hours']
    pi = pd.date_range(end=decision, periods=cfg_a['history_days'] * 24 // step, freq=f'{step}h')
    fi_end = decision + pd.Timedelta(hours=cfg_a['horizon_days'] * 24)
    years = sorted(set(pi.year) | set(pd.date_range(decision, fi_end, freq='1D').year))
    out_dir = os.path.join(OUT, day + ('_check' if a.check else ''))
    inputs = os.path.join(out_dir, 'inputs')
    os.makedirs(inputs, exist_ok=True)
    log(f'起报 {issue:%Y-%m-%d} 00 UTC，决策时刻 {decision:%m-%d %H} UTC；设备 {device} → {os.path.relpath(out_dir, ROOT)}')
    stations = dl.station_list(a.sites)
    rf, gefs_note = gefs_issue(day, stations, a, inputs, bases)

    # 3. past 7 days and the per-site datasets
    h = {sid: harmonization(sid, a.harmonization) for sid in a.sites}
    buoy = {sid: buoy_hs(sid, prep.SITES[sid], pi, a.obs) for sid in a.sites}
    era5 = {sid: era5_context(sid, pi) for sid in a.sites}
    need_recent = not a.check and any(era5[s] is None or buoy[s][0].isna().any() for s in a.sites)
    recent = gefs_recent(day, stations, a, inputs, bases, rf) if need_recent else {}
    info = {sid: write_site(sid, prep.SITES[sid], rf, h[sid], pi, buoy[sid], era5[sid], recent, a, inputs) for sid in a.sites}
    for sid, i in info.items():
        log(f'{sid}：过去 Hs ← {i["past_hs"]}；过去 Te / 波向 ← {i["past_te_mwd"]}' + (f'；不预报：{i["skip"]}' if 'skip' in i else ''))
        for n in i['notes']:
            log(f'    {n}')
    sids = [s for s in a.sites if 'skip' not in info[s]]
    if not sids:
        raise SystemExit('没有可以预报的站')

    # 4. predictions
    q, keys = {}, None
    for run, (cfg, model, abl) in runs.items():
        smp, skipped, names, _ = samples_for(cfg, abl, sids, inputs, years, decision)
        if len(smp) != len(sids):
            raise SystemExit(f'应有 {len(sids)} 个站点样本，实际 {len(smp)} 个（跳过原因 {skipped}）')
        k = [(m['site_id'], m['valid_time']) for s in smp for m in s['metadata']]
        if keys is not None and k != keys:
            raise SystemExit('集成成员的样本行不一致')
        keys, samples = k, smp
        q[model], _ = predict_seeds(cfg, model, abl, smp, device)
        log(f'预测 {run}：{len(smp)} 个站 × {len(smp[0]["y"])} 个时刻')
    qe = np.mean(list(q.values()), axis=0)
    tq = None
    if te_run:
        cfg_t, model_t, abl_t = te_run
        smp_t, _, _, _ = samples_for(cfg_t, abl_t, sids, inputs, years, decision)
        if [(m['site_id'], m['valid_time']) for s in smp_t for m in s['metadata']] != keys:
            raise SystemExit('周期模型的样本行与波高模型不一致')
        _, tq = predict_seeds(cfg_t, model_t, abl_t, smp_t, device)
        log(f'预测周期模型 {TE_RUN}')

    # 5. table
    meta = pd.DataFrame([m for s in samples for m in s['metadata']])
    out = meta[['site_id', 'issue_time', 'decision_time', 'valid_time', 'lead_hours', 'lead_day']].copy()
    out['site_id'] = out['site_id'].astype(str)
    sites_train, _ = load_sites(cfg_a)
    clim = site_climatology([s for s in sites_train if str(s[0]['site_id']) in sids])
    # 10-29：模型最后 --blend-days 天按分位数逐步过渡到气候态（验证集上比较 0/2/3/5 天，第 12–15 天 CRPS 都不变差、N = 2 最低，
    # 衔接跳变平均 0.17 → 0.01 m；compare_blend_days.py）。--check 比对的仍是未过渡的模型输出 qe
    w = blend_weight(out['lead_hours'], cfg_a, a.blend_days)[:, None]
    doy = pd.to_datetime(out['valid_time'], utc=True).dt.dayofyear.to_numpy()
    qp = (1 - w) * qe + w * np.stack([clim[s][d]['hs'] for s, d in zip(out['site_id'], doy)])
    for p_, col in LEVELS:
        out[f'Hs_{col}'] = quantile_at(qp, p_)
    for t in a.threshold:
        out[f'P_Hs_gt_{t:g}m'] = 1 - np.array([np.interp(t, row, TAU, left=0, right=1) for row in qp])
    raw = np.concatenate([s['raw_quantiles'] for s in samples])
    for p_, col in [(.1, 'P10'), (.5, 'P50'), (.9, 'P90')]:
        out[f'GEFS_raw_{col}'] = quantile_at(raw, p_)
    out['GEFS_Te_mean_s'] = np.concatenate([s['future'][:, names.index('te')] for s in samples])
    sids_row, vt = out['site_id'].to_numpy(), pd.to_datetime(out['valid_time'], utc=True)
    measured = {s for s in sids if glob.glob(os.path.join(ROOT, 'processed', 'buoy_spectral', f'{s}_*.csv'))}
    use = np.isin(sids_row, list(measured)) & (tq is not None)
    te_q = np.where(use[:, None], tq if tq is not None else 0, out['GEFS_Te_mean_s'].to_numpy()[:, None])
    for p_, col in [(.1, 'P10'), (.5, 'P50'), (.9, 'P90')]:
        out[f'Te_{col}_s'] = quantile_at(te_q, p_)
    out['Te_source'] = np.where(use, 'corrected', 'GEFS_raw')
    from joint_scenarios import copula_rho, joint_samples      # 需要 scipy
    rho = np.zeros(len(out))
    table = pd.read_csv(a.copula_table, dtype={'site_id': str}) if os.path.exists(a.copula_table) else None
    te_v, _ = observed_te(sites_train)
    for c, o, _ in sites_train:
        sid = str(c['site_id'])
        if sid not in sids:
            continue
        if table is not None and (table.site_id == sid).any():
            r = {int(m): float(v) for m, v in table[table.site_id == sid][['month', 'rho']].values.tolist()}
        else:
            train = o.index.year.isin(c['train_years'])
            r = copula_rho(o['hs'][train], te_v[sid].reindex(o.index)[train])
        k = sids_row == sid
        rho[k] = [r[m] for m in vt[k].dt.month]
    hs_s, te_s = joint_samples(qp, te_q, rho, n=200)
    wef = FLUX * hs_s ** 2 * te_s
    for p_, col in [(.1, 'P10'), (.5, 'P50'), (.9, 'P90')]:
        out[f'WEF_{col}_kWm'] = np.quantile(wef, p_, axis=1)
    out['WEF_mean_kWm'] = wef.mean(1)
    out['source'] = np.where(w[:, 0] > 0, 'model_blend', 'model')
    if a.days > cfg_a['horizon_days']:
        out['extreme_at_issue'] = ''
        ext = climatology_rows(out, sites_train, cfg_a, a.days, clim)
        doy = pd.to_datetime(ext['valid_time'], utc=True).dt.dayofyear.to_numpy()
        hq = np.stack([clim[s][d]['hs'] for s, d in zip(ext['site_id'].astype(str), doy)])
        for t in a.threshold:
            ext[f'P_Hs_gt_{t:g}m'] = 1 - np.array([np.interp(t, row, TAU, left=0, right=1) for row in hq])
        ext = ext.drop(columns=[c for c in ext if 'observed' in c or c == 'Te_obs_source'])
        out = pd.concat([out, ext], ignore_index=True).drop(columns='extreme_at_issue')
        out = out.sort_values(['site_id', 'lead_hours'], kind='stable').reset_index(drop=True)
        add_period_energy(out, clim, a.days)
    out['decision_time'] = pd.to_datetime(out['decision_time'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    for c in ('issue_time', 'valid_time'):
        out[c] = pd.to_datetime(out[c], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    path = os.path.join(out_dir, f'forecast_{a.days}d_{stamp}.csv')
    out.round(3).to_csv(path, index=False, encoding='utf-8-sig')
    log(f'预报表 {len(out)} 行 → {os.path.relpath(path, ROOT)}')

    result = dict(issue=f'{issue:%Y-%m-%dT%H:%M:%SZ}', decision=f'{decision:%Y-%m-%dT%H:%M:%SZ}', run_time=t_run.isoformat(timespec='seconds'),
                  device=device, gefs=gefs_note, sites=info, harmonization=h, threshold_m=a.threshold,
                  runs={r: dict(model=m, ablation=ab, seeds=c['seeds'],
                                config_hash_portable=json.load(open(os.path.join(c['output_dir'], 'selection.json'), encoding='utf-8')).get('config_hash_portable'))
                        for r, (c, m, ab) in list(runs.items()) + ([(TE_RUN, te_run)] if te_run else [])},
                  ensemble=len(a.runs) == len(RUNS), blend_days=a.blend_days, output=os.path.relpath(path, ROOT))
    if a.check:
        result['check'] = check(day, a.runs[0], q, qe, keys)
    json.dump(result, open(os.path.join(out_dir, f'run_{stamp}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    write_summary(out, info, result, a, os.path.join(out_dir, f'summary_{stamp}.md'), cfg_a)


def check(day, run_a, q, qe, keys):
    """--check：与业务独立检验保存的预测逐行比对（同一起报、同一输入，差应在浮点误差量级）。"""
    d = os.path.join(ROOT, 'results', 'operational', f'{run_a}_ens_oper')
    z, cases = np.load(os.path.join(d, 'quantiles.npz')), pd.read_csv(os.path.join(d, 'cases.csv'))
    cases = cases[pd.to_datetime(cases['issue_time'], utc=True) == pd.Timestamp(day, tz='UTC')]
    pos = {(str(s), pd.Timestamp(v)): i for s, v, i in zip(cases['site_id'], pd.to_datetime(cases['valid_time'], utc=True), cases.index)}
    idx = [pos.get((str(s), pd.Timestamp(v))) for s, v in keys]
    if None in idx:
        log(f'核对：{idx.count(None)} 个时刻在业务检验里没有（那里只保留有实测的时刻），只比对其余 {len(idx) - idx.count(None)} 个')
    k = [j for j, i in enumerate(idx) if i is not None]
    rows = np.array([idx[j] for j in k])
    res = {}
    for name, v in list(q.items()) + ([('ensemble', qe)] if 'ensemble' in z.files and len(q) == len(RUNS) else []):
        diff = float(np.abs(v[k] - z[name][rows]).max())
        res[name] = dict(max_abs_diff_m=diff, ok=diff < 1e-3)
        log(f'核对 {name}：与业务检验保存的预测最大差 {diff:.2e} m → {"通过" if diff < 1e-3 else "不通过"}')
    return res


def write_summary(out, info, result, a, path, cfg):
    thr = a.threshold[0]
    lines = [f'# 30 天波浪概率预报 · 起报 {result["issue"][:10]} 00 UTC', '',
             f'运行时刻 {result["run_time"]}；决策时刻 {result["decision"]}；模型：' +
             ('Transformer + GRU 集成（冻结）' if result['ensemble'] else '**仅部分集成成员（测试用）**') +
             f'；第 1–{cfg["horizon_days"]} 天模型，之后气候态'
             + (f'（第 {cfg["horizon_days"] - a.blend_days + 1}–{cfg["horizon_days"]} 天逐步过渡）' if a.blend_days else '') + '。', '']
    if 'check' in result:
        lines += ['## 历史起报核对', '', '| 预测 | 最大差 (m) | 结果 |', '|---|---:|---|']
        lines += [f'| {k} | {v["max_abs_diff_m"]:.2e} | {"通过" if v["ok"] else "**不通过**"} |' for k, v in result['check'].items()] + ['']
    lines += ['## 输入来源', '', '| 站 | 过去 7 天 Hs | 过去 Te / 波向 | 说明 |', '|---|---|---|---|']
    for sid, i in info.items():
        lines.append(f'| {sid} | {i["past_hs"]} | {i["past_te_mwd"]} | {"；".join(i["notes"] + ([i["skip"]] if "skip" in i else [])) or "—"} |')
    lines += ['', f'GEFS 这次起报：{result["gefs"]}', '']
    pcol = f'P_Hs_gt_{thr:g}m'
    for sid in [s for s in a.sites if (out['site_id'] == s).any()]:
        g = out[out['site_id'] == sid]
        g = g.assign(day=pd.to_datetime(g['valid_time']).dt.strftime('%m-%d'))
        lines += [f'## {sid} 站', '', f'| 预见日 | 日期 | P10 (m) | P50 (m) | P90 (m) | 超过 {thr:g} m 的概率 | 来源 |', '|---:|---|---:|---:|---:|---:|---|']
        for d, gd in g.groupby('lead_day'):
            lines.append(f'| {d} | {gd["day"].iloc[0]} | {gd.Hs_P10.mean():.2f} | {gd.Hs_P50.mean():.2f} | {gd.Hs_P90.max():.2f} | '
                         f'{gd[pcol].max():.0%} | {"模型" if (gd.source == "model").all() else "模型→气候态过渡" if gd.source.str.startswith("model").all() else "气候态"} |')
        col = f'WEF_next{a.days}d_mean_kWm'
        if col in g:
            r = g.iloc[0]
            lines += ['', f'未来 {a.days} 天平均波能流 {r[col]:.1f} kW/m（前 15 天 {r["WEF_next15d_mean_kWm"]:.1f}），'
                          f'同期气候态 {r[f"WEF_next{a.days}d_clim_kWm"]:.1f}，距平 {r[f"WEF_next{a.days}d_anomaly_pct"]:+.0f}%。']
        lines.append('')
    lines += ['## 怎么读', '',
              '- P10 / P50 / P90：当天各时刻 P10、P50 的平均和 P90 的最大值；实测有 80% 的把握落在 P10–P90 之间。',
              f'- 超过 {thr:g} m 的概率：当天各时刻中的最大值，由 99 个分位数直接读出。',
              '- 大浪时预报上尾偏轻（业务检验最高一成条件 PIT 0.18，理想 0.10）：安全判断宜保守，可参考 P95。',
              f'- 第 {cfg["horizon_days"] + 1}–{a.days} 天是气候态（训练年同日期 ±15 天的实测分布），只代表历史同期的正常范围，不预报具体天气过程。',
              '- 过去 7 天输入由 GEFS 代替浮标时（见"输入来源"），验证集上整体 CRPS 约变差 1%，关岛约 2%。', '']
    open(path, 'w', encoding='utf-8').write('\n'.join(lines))
    log(f'一览 → {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
