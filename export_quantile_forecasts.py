"""把概率预报导出成表格：每个站点、每次起报、每个预见时刻的 P05 / P10 / P25 / P50 / P75 / P90 / P95（有效波高 Hs，m）。

同时附上：实测 Hs（有真值时）、原始 GEFS 集合的 P10 / P50 / P90、GEFS 集合平均能量周期 Te（s，供最后的功率矩阵收尾用）、
起报时是否有极端天气在附近（台风 / 寒潮）；波能流 WEF（kW/m，= 0.49·Hs²·Te）的 P10 / P50 / P90 / 均值，
以及"实测"波能流（浮标 Hs + 实测 Te：有 CDIP 海浪谱的站（关岛 G、夏威夷 H）用谱算的真实 Te，
其余站（马罗岛 K）用 ERA5 Te，Te_obs_source 列注明；10-23 起，之前一律用 ERA5 Te，夏威夷因此偏高约 10%）。月平均波浪能与 MVI 由 forecast_wave_energy_mvi.py 汇总。
30 天产品（前 15 天模型、后 15 天气候态）：--days 30 时，模型时效之后到第 30 天的时刻用气候态补齐
（source = climatology）：Hs 分位数 = 该站训练年份浮标实测、预见日期 ±15 天的分位数（不用验证 / 测试年份；
10-23 在验证集上比过：比按日历月好 G −2.2%、K −1.1%，H 持平，见 climatology_days16_30.py）；
波能流分位数和均值 = 训练年份"浮标 Hs + ERA5 Te"波能流在同一日期窗内的分位数和均值。--blend-days N：模型最后 N 天按分位数
线性加权逐步过渡到气候态（权重从 0 升到接近 1），避免模型与气候态交界处跳变；默认 0（直接切换），是否过渡要在验证集上比较后再定。
用法：
  python export_quantile_forecasts.py --split test                 最终模型（调参后 Transformer）的测试集预报（已评过分）
  python export_quantile_forecasts.py --split val --variant ensemble_specialist
        验证集上"集成 + 预训练 + 极端专用"的预报（要先跑完两个模型的 train_extreme_specialist.py）
  python export_quantile_forecasts.py --split val --config GKH_h15_transformer_pretrain --days 30
  python export_quantile_forecasts.py --split val --config GKH_h15_transformer_pretrain --ensemble-with GKH_h15_gru_pretrain --days 30
        10-27 起的最终方案：Transformer + GRU 分位数平均（文件名加 _ens）
        任一训练好的配置（验证集取各种子平均；测试集读 <output_dir>/test/quantiles.npz），补到 30 天
  python export_quantile_forecasts.py --split val --config GKH_h15_transformer_pretrain --te-config GKH_h15_te_transformer_pretrain --days 30
        10-23 加周期：--te-config 指定多任务模型（周期头），有实测 Te 的站（关岛 G、夏威夷 H）输出订正后的 Te 分位数，
        其余站（马罗岛 K，没有实测 Te 可检验）用 GEFS 原始 Te（Te_source 列注明）；波能流由 Hs–Te 成对情景
        （joint_scenarios.py，高斯 copula，相关取训练年实测）算 P10 / P50 / P90 / 均值，旧算法（GEFS Te × Hs 分位数）
        的均值保留在 WEF_mean_gefsTe_kWm 供比较；--scenarios N 另存 N 个成对情景（npz，功率矩阵用）
输出：results/forecast_outputs/<方案或配置>_<split>[_30d].csv（UTF-8，Excel 可直接打开）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from project_paths import resolve_config                       # noqa: E402
from wavecast.data import feature_names                         # noqa: E402
from wavecast.experiment import load_sites, build_all           # noqa: E402
from wavecast.metrics import quantile_at, TAU                   # noqa: E402

OUT = os.path.join(ROOT, 'results', 'forecast_outputs')
FLUX = 1025.0 * 9.81 ** 2 / (64 * np.pi) / 1000          # 深水波能流系数 ≈ 0.49 kW/(m³·s)
LEVELS = [(.05, 'P05'), (.10, 'P10'), (.25, 'P25'), (.50, 'P50'), (.75, 'P75'), (.90, 'P90'), (.95, 'P95')]


def cfg_of(name):
    return json.load(open(resolve_config(f'model_configs/{name}.json'), encoding='utf-8'))


def quantiles_of_config(name, split, model=None):
    c = cfg_of(name)
    model = model or c.get('models', ['transformer'])[0]
    ablation = json.load(open(os.path.join(c['output_dir'], 'selection.json'), encoding='utf-8'))['best'][model]['ablation']
    if split == 'test':
        path = os.path.join(c['output_dir'], 'test', 'quantiles.npz')
        if not os.path.exists(path):
            raise SystemExit(f'{name} 没有测试集结果（测试集只评一次，要先确定最终模型）')
        return c, ablation, np.load(path)[model]
    return c, ablation, np.mean([np.load(os.path.join(c['output_dir'], f'{model}_{ablation}_seed{s}_val.npz'))['quantiles']
                                 for s in c['seeds']], axis=0)


def quantiles(variant, split):
    if split == 'test':
        if variant != 'final':
            raise SystemExit('测试集只导出最终模型（其他方案没有在测试集上评价）')
        c = cfg_of('GKH_reforecast_joint_tw_tuned_transformer')
        return c, np.load(os.path.join(c['output_dir'], 'test', 'quantiles.npz'))['transformer']
    if variant == 'final':
        c = cfg_of('GKH_reforecast_joint_tw_tuned_transformer')
        return c, np.mean([np.load(os.path.join(c['output_dir'], f'transformer_season_seed{s}_val.npz'))['quantiles']
                           for s in c['seeds']], axis=0)
    if variant == 'ensemble_specialist':
        ct, cg = cfg_of('GKH_tuned_transformer_pretrain'), cfg_of('GKH_tuned_gru_pretrain')
        qt = np.load(os.path.join(ct['output_dir'], 'extreme_specialist', 'routed_transformer_val.npz'))['quantiles']
        qg = np.load(os.path.join(cg['output_dir'], 'extreme_specialist', 'routed_gru_val.npz'))['quantiles']
        return ct, (qt + qg) / 2
    raise SystemExit(f'未知方案 {variant}')


def observed_te(sites):
    """Te 'truth' per site on each site's observation times: buoy spectral Te (processed/buoy_spectral, CDIP) where it
    exists, otherwise ERA5 Te. Returns (values, source) frames with one column per site."""
    vals, src = {}, {}
    for c, o, _ in sites:
        sid = str(c['site_id'])
        era5 = o['te'].copy()
        era5.index = era5.index if era5.index.tz is not None else era5.index.tz_localize('UTC')
        files = glob.glob(os.path.join(ROOT, 'processed', 'buoy_spectral', f'{sid}_*.csv'))
        buoy = pd.Series(dtype=float)
        if files:
            b = pd.read_csv(files[0], usecols=['valid_time', 'te'], parse_dates=['valid_time']).set_index('valid_time').te
            buoy = b.reindex(era5.index)
        has = buoy.notna() if len(buoy) else pd.Series(False, index=era5.index)
        vals[sid] = buoy.where(has, era5) if len(buoy) else era5
        src[sid] = pd.Series(np.where(has, 'buoy', 'ERA5'), index=era5.index)
    return pd.DataFrame(vals), pd.DataFrame(src)


def site_climatology(sites, window=15):
    """Per site and day of year (training years only, ±window days, 10-23 chosen on validation over calendar months):
    Hs quantiles at TAU, wave-energy-flux quantiles and mean."""
    out = {}
    for c, obs, _ in sites:
        years = c['train_years']
        if obs.loc[obs.index.year.isin(years), 'hs'].notna().sum() < 365 * 8:
            # 10-26：浮标晚于训练年份才开始的站（如韩国 22185 等 2015-12 起）：改用验证、测试年份以外的年份（不参与评分）
            banned = set(c['val_years']) | set(c['test_years'])
            years = sorted(set(obs.index.year) - banned)
            print(f'  {c["site_id"]}：训练年份没有足够实测，气候态改用验证 / 测试以外的年份 {years[0]}–{years[-1]}', flush=True)
        train = obs.loc[obs.index.year.isin(years)]
        te = observed_te([(c, obs, None)])[0].iloc[:, 0].to_numpy()[obs.index.year.isin(years)]
        hs, wef = train['hs'], FLUX * train['hs'] ** 2 * te
        te_s = pd.Series(te, index=train.index)
        dy = train.index.dayofyear.to_numpy()
        table = {}
        for d in range(1, 367):
            near = np.minimum(abs(dy - d), 366 - abs(dy - d)) <= window
            h, w = hs[near].dropna(), wef[near].dropna()
            if len(h) < 20 or len(w) < 20:
                raise SystemExit(f'站点 {c["site_id"]} 第 {d} 天附近训练年份实测太少，无法做气候态')
            t = te_s[near].dropna()
            table[d] = dict(hs=np.quantile(h, TAU), wef_q=np.quantile(w, [.1, .5, .9]), wef_mean=w.mean(),
                            te_q=np.quantile(t, [.1, .5, .9]) if len(t) >= 20 else np.full(3, np.nan))
        out[str(c['site_id'])] = table
    return out


def observed(sites, column):
    t = pd.concat([o[column].rename(str(c['site_id'])) for c, o, _ in sites], axis=1)
    t.index = t.index if t.index.tz is not None else t.index.tz_localize('UTC')
    return t


def lookup(table, times, site_ids):
    return np.array([table.at[t, s] if t in table.index else np.nan for t, s in zip(times, site_ids)], dtype=float)


def climatology_rows(out, sites, cfg, days, clim):
    """Rows after the model horizon up to `days` days after the decision time, filled with climatology."""
    step = cfg['step_hours']
    first, last = cfg['horizon_days'] * 24 // step + 1, days * 24 // step
    heads = out.drop_duplicates(['site_id', 'issue_time'])[['site_id', 'issue_time', 'decision_time', 'extreme_at_issue']]
    rows = []
    for _, h in heads.iterrows():
        decision = pd.Timestamp(h.decision_time)
        for k in range(first, last + 1):
            rows.append(dict(site_id=h.site_id, issue_time=h.issue_time, decision_time=h.decision_time,
                             valid_time=decision + pd.Timedelta(hours=k * step), lead_hours=k * step,
                             lead_day=(k - 1) * step // 24 + 1, extreme_at_issue=h.extreme_at_issue))
    ext = pd.DataFrame(rows)
    vt = pd.to_datetime(ext.valid_time, utc=True)
    months, sids = vt.dt.dayofyear.to_numpy(), ext.site_id.astype(str).to_numpy()   # 日序（变量名沿用）
    hq = np.stack([clim[s][m]['hs'] for s, m in zip(sids, months)])
    for p_, col in LEVELS:
        ext[f'Hs_{col}'] = quantile_at(hq, p_)
    ext['Hs_observed'] = lookup(observed(sites, 'hs'), vt, sids)
    for i, col in enumerate(('P10', 'P50', 'P90')):
        ext[f'WEF_{col}_kWm'] = [clim[s][m]['wef_q'][i] for s, m in zip(sids, months)]
    ext['WEF_mean_kWm'] = [clim[s][m]['wef_mean'] for s, m in zip(sids, months)]
    for i, col in enumerate(('P10', 'P50', 'P90')):
        ext[f'Te_{col}_s'] = [clim[s][m]['te_q'][i] for s, m in zip(sids, months)]
    ext['Te_source'] = "climatology"
    te_v, te_s = observed_te(sites)
    ext['Te_observed_s'] = lookup(te_v, vt, sids)
    ext['Te_obs_source'] = [te_s.at[t, s] if t in te_s.index else '' for t, s in zip(vt, sids)]
    ext['WEF_observed_kWm'] = FLUX * ext.Hs_observed ** 2 * ext.Te_observed_s
    ext['source'] = 'climatology'
    return ext


def add_period_energy(out, clim, days):
    """Per site × issue: mean wave-energy flux over the next `days` and next 15 days (forecast distribution mean;
    days after the model horizon are climatology), the same-period climatology and the anomaly. Validation (10-26):
    the 30-day mean beats climatology (skill +0.15–0.20, anomaly correlation 0.44–0.61), all of it from days 1–15."""
    doy = pd.to_datetime(out.valid_time, utc=True).dt.dayofyear.to_numpy()
    out['_clim'] = [clim[str(s)][d]['wef_mean'] for s, d in zip(out.site_id, doy)]
    g = out.groupby(['site_id', 'issue_time'], sort=False)
    first15 = out.lead_day <= 15
    out[f'WEF_next{days}d_mean_kWm'] = g['WEF_mean_kWm'].transform('mean')
    out['WEF_next15d_mean_kWm'] = out['WEF_mean_kWm'].where(first15).groupby([out.site_id, out.issue_time]).transform('mean')
    out[f'WEF_next{days}d_clim_kWm'] = g['_clim'].transform('mean')
    out[f'WEF_next{days}d_anomaly_pct'] = (out[f'WEF_next{days}d_mean_kWm'] / out[f'WEF_next{days}d_clim_kWm'] - 1) * 100
    out.drop(columns='_clim', inplace=True)


def te_quantile_rows(te_config, split, n_rows):
    """Seed-averaged Te quantiles of the multi-task model (validation), or its test-set file once it exists."""
    c = cfg_of(te_config)
    model = c.get('models', ['transformer'])[0]
    ablation = json.load(open(os.path.join(c['output_dir'], 'selection.json'), encoding='utf-8'))['best'][model]['ablation']
    path = (os.path.join(c['output_dir'], f'{model}_{ablation}_val_te_mean.npz') if split == 'val'
            else os.path.join(c['output_dir'], 'test', 'te_quantiles.npz'))
    if not os.path.exists(path):
        raise SystemExit(f'没有周期分位数文件 {path}')
    tq = np.load(path)['quantiles']
    if len(tq) != n_rows:
        raise SystemExit(f'周期模型的行数 {len(tq)} 与波高模型 {n_rows} 不一致（两个配置的站点、年份、时效必须相同）')
    return tq


def joint_energy(out, q, a, sites, smp, vt, sids, te_v, label):
    """Te columns and wave-energy flux from Hs–Te joint scenarios (corrected Te where measured Te exists)."""
    from joint_scenarios import copula_rho, joint_samples      # 需要 scipy；只在 --te-config 时加载
    tq = te_quantile_rows(a.te_config, a.split, len(out))
    measured = {str(c['site_id']) for c, _, _ in sites
                if glob.glob(os.path.join(ROOT, 'processed', 'buoy_spectral', f"{c['site_id']}_*.csv"))}
    use = np.isin(sids, list(measured))
    raw = out['GEFS_Te_mean_s'].to_numpy()
    tq = np.where(use[:, None], tq, raw[:, None])                 # 没有实测 Te 的站：GEFS 原始 Te（单值）
    for p_, col in [(.1, 'P10'), (.5, 'P50'), (.9, 'P90')]:
        out[f'Te_{col}_s'] = quantile_at(tq, p_)
    out['Te_source'] = np.where(use, 'corrected', 'GEFS_raw')
    rho = np.zeros(len(out))
    pit_table = pd.read_csv(a.copula_table, dtype={'site_id': str}) if a.copula_table else None
    for c, o, _ in sites:
        sid = str(c['site_id'])
        train = o.index.year.isin(c['train_years'])
        if pit_table is not None and (pit_table.site_id == sid).any():
            table = dict(pit_table[pit_table.site_id == sid][['month', 'rho']].values.tolist())
            table = {int(m): float(r) for m, r in table.items()}
        else:
            table = copula_rho(o['hs'][train], te_v[sid].reindex(o.index)[train])
        k = sids == sid
        rho[k] = [table[m] for m in vt[k].dt.month]
        print(f'  {sid}：Hs–Te 高斯相关（{"条件相关表" if pit_table is not None and (pit_table.site_id == sid).any() else "训练年实测"}，按月）{min(table.values()):.2f} – {max(table.values()):.2f}', flush=True)
    hs_s, te_s = joint_samples(np.asarray(q), tq, rho, n=200)
    wef = FLUX * hs_s ** 2 * te_s
    out['WEF_mean_gefsTe_kWm'] = out['WEF_mean_kWm']
    for p_, col in [(.1, 'P10'), (.5, 'P50'), (.9, 'P90')]:
        out[f'WEF_{col}_kWm'] = np.quantile(wef, p_, axis=1)
    out['WEF_mean_kWm'] = wef.mean(1)
    if a.scenarios:
        k = np.linspace(0, 199, a.scenarios).round().astype(int)
        path = os.path.join(OUT, f'{label}_{a.split}' + ('_pitcop' if a.copula_table else '') + '_scenarios.npz')
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(path, hs=hs_s[:, k].astype(np.float32), te=te_s[:, k].astype(np.float32),
                            site_id=sids, valid_time=vt.dt.strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy(),
                            issue_time=pd.to_datetime(out.issue_time, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy())
        print(f'  成对情景 {a.scenarios} 个 → {path}', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--variant', choices=['final', 'ensemble_specialist'], default='final')
    p.add_argument('--config', help='导出这个配置训练好的模型（model_configs 里的名字，不带 .json）')
    p.add_argument('--model', help='配置里有多个模型时指定一个')
    p.add_argument('--ensemble-with', help='与另一个配置（如 GKH_h15_gru_pretrain）的分位数等权平均（集成，10-27 起的最终方案）；'
                                            '两个配置的数据与时效必须相同')
    p.add_argument('--days', type=int, default=0, help='补到第几天（如 30）；模型时效之后用气候态')
    p.add_argument('--blend-days', type=int, default=0, help='模型最后几天逐步过渡到气候态（默认 0 = 直接切换）')
    p.add_argument('--te-config', help='周期多任务模型的配置名（输出订正后的 Te 与成对情景波能流）')
    p.add_argument('--scenarios', type=int, default=0, help='另存多少个 Hs–Te 成对情景（如 50）')
    p.add_argument('--copula-table', help='条件相关表（copula_from_pit.py 的输出）；不给则用训练年实测的无条件相关')
    p.add_argument('--conformal', help='共形校正平移量 JSON（postproc_diagnostics.py 的输出，按 站点|预见期段）；'
                                         '只用验证集定的平移量，导出测试集 / 业务预报时加上')
    a = p.parse_args()
    if a.config:
        cfg, ablation, q = quantiles_of_config(a.config, a.split, a.model)
        label = a.config
        if a.ensemble_with:
            c2, _, q2 = quantiles_of_config(a.ensemble_with, a.split)
            if q2.shape != q.shape or c2['horizon_days'] != cfg['horizon_days']:
                raise SystemExit(f'{a.ensemble_with} 与 {a.config} 的预报行数或时效不同，不能集成')
            # quantile averaging (Vincentization) of the two backbones, as in route_models.py
            q = (np.asarray(q) + np.asarray(q2)) / 2
            label += '_ens'
    else:
        cfg, q = quantiles(a.variant, a.split)
        ablation, label = 'season', a.variant
    sites, _ = load_sites(cfg)
    names = feature_names(cfg, ablation)
    samples, _, names = build_all(sites, cfg, names)
    smp = samples[a.split]
    if len(q) != sum(len(s['y']) for s in smp):
        raise SystemExit('预报行数与样本不一致')
    from train_extreme_specialist import extreme_flags
    flags, kinds = extreme_flags(sites, smp, cfg, 1000)
    meta = pd.DataFrame([m for s in smp for m in s['metadata']])
    out = meta[['site_id', 'issue_time', 'decision_time', 'valid_time', 'lead_hours', 'lead_day']].copy()
    out['extreme_at_issue'] = np.repeat(kinds, [len(s['y']) for s in smp])
    if a.conformal:
        from wavecast.postproc import apply_shifts
        shifts = {g: np.asarray(v) for g, v in json.load(open(a.conformal, encoding='utf-8'))['shifts'].items()}
        grp = (meta.site_id.astype(str) + '|' + meta.lead_band).to_numpy()
        q = apply_shifts(q, grp, shifts)
        label += '_cqr'
        print(f'  共形校正：{np.isin(grp, list(shifts)).mean():.0%} 的行有对应平移量（其余不变）')
    clim = site_climatology(sites) if a.days or a.blend_days else None
    w = np.zeros((len(out), 1))
    if a.blend_days:
        # 模型最后 blend_days 天逐步过渡到气候态：分位数线性加权（Vincentization），权重 0 → 接近 1
        steps = a.blend_days * 24 // cfg['step_hours']
        horizon = cfg['horizon_days'] * 24 // cfg['step_hours']
        j = out.lead_hours.to_numpy() // cfg['step_hours']
        w = np.clip((j - (horizon - steps)) / (steps + 1), 0, 1)[:, None]
        months = pd.to_datetime(out.valid_time, utc=True).dt.dayofyear.to_numpy()
        hq = np.stack([clim[str(s)][m]['hs'] for s, m in zip(out.site_id, months)])
        q = (1 - w) * np.asarray(q) + w * hq
    for p_, col in LEVELS:
        out[f'Hs_{col}'] = quantile_at(q, p_)
    out['Hs_observed'] = np.concatenate([s['y'] for s in smp])
    raw = np.concatenate([s['raw_quantiles'] for s in smp])
    for p_, col in [(.1, 'P10'), (.5, 'P50'), (.9, 'P90')]:
        out[f'GEFS_raw_{col}'] = quantile_at(raw, p_)
    if 'te' in names:
        out['GEFS_Te_mean_s'] = np.concatenate([s['future'][:, names.index('te')] for s in smp])
        # 波能流 P = 0.49·Hs²·Te（kW/m）：周期暂用 GEFS 集合平均 Te，所以 P 的分位数 = 0.49·Te·(Hs 分位数)²；
        # 均值用 E[Hs²] = 99 个等概率分位数平方的平均（月平均、MVI 要用均值，不能用中位数）
        te = out['GEFS_Te_mean_s'].to_numpy()
        for col in ('P10', 'P50', 'P90'):
            out[f'WEF_{col}_kWm'] = FLUX * out[f'Hs_{col}'] * out[f'Hs_{col}'] * te
        out['WEF_mean_kWm'] = FLUX * (np.asarray(q) ** 2).mean(axis=1) * te
        vt = pd.to_datetime(out.valid_time, utc=True)
        sids = out.site_id.astype(str).to_numpy()
        te_v, te_s = observed_te(sites)
        if a.te_config:
            joint_energy(out, q, a, sites, smp, vt, sids, te_v, label)
        out['Te_observed_s'] = lookup(te_v, vt, sids)
        out['Te_obs_source'] = [te_s.at[t, s] if t in te_s.index else '' for t, s in zip(vt, sids)]
        out['WEF_observed_kWm'] = FLUX * out.Hs_observed ** 2 * out.Te_observed_s
    out['source'] = np.where(w[:, 0] > 0, 'model_blend', 'model')
    n_runs = len(out.drop_duplicates(['site_id', 'issue_time']))
    n_model = len(out) // max(n_runs, 1)
    if a.days:
        if a.days <= cfg['horizon_days']:
            raise SystemExit(f'--days {a.days} 不超过模型时效 {cfg["horizon_days"]} 天，不需要气候态')
        out = pd.concat([out, climatology_rows(out, sites, cfg, a.days, clim)], ignore_index=True)
        out = out.sort_values(['site_id', 'issue_time', 'lead_hours'], kind='stable')
        if 'WEF_mean_kWm' in out:
            add_period_energy(out, clim, a.days)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{label}_{a.split}' + ('_pitcop' if a.copula_table else '') + (f'_{a.days}d' if a.days else '') + '.csv')
    out.round(3).to_csv(path, index=False, encoding='utf-8-sig')
    col = f'WEF_next{a.days}d_mean_kWm'
    if a.days and col in out:
        # 每次起报一行的时段平均波浪能表（未来 a.days 天 / 15 天平均、同期气候态、距平；有实测时附实测时段平均）
        summary = out.groupby(['site_id', 'issue_time'], sort=False).agg(
            decision_time=('decision_time', 'first'), **{col: (col, 'first'),
            'WEF_next15d_mean_kWm': ('WEF_next15d_mean_kWm', 'first'),
            f'WEF_next{a.days}d_clim_kWm': (f'WEF_next{a.days}d_clim_kWm', 'first'),
            f'WEF_next{a.days}d_anomaly_pct': (f'WEF_next{a.days}d_anomaly_pct', 'first'),
            f'WEF_observed_next{a.days}d_kWm': ('WEF_observed_kWm', 'mean'),
            'observed_fraction': ('WEF_observed_kWm', lambda x: x.notna().mean())}).reset_index()
        spath = path.replace('.csv', '_period_energy.csv')
        summary.round(3).to_csv(spath, index=False, encoding='utf-8-sig')
        print(f'  时段平均波浪能（每次起报一行）{len(summary)} 行 → {spath}')
    for src, g in out.groupby('source'):
        g = g.dropna(subset=['Hs_observed'])
        cov = ((g.Hs_observed >= g.Hs_P10) & (g.Hs_observed <= g.Hs_P90)).mean()
        print(f'  {src}：{len(g)} 个有实测的时刻，P10–P90 实际覆盖率 {cov:.1%}')
    print(f'导出 {len(out)} 行（{n_runs} 个站点起报 × 模型 {n_model} 个时刻'
          + (f'，气候态补到第 {a.days} 天' if a.days else '') + f'）→ {path}')


if __name__ == '__main__':
    main()
