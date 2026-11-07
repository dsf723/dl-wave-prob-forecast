"""月内日变化指数：未来 30 天（及前 15 天）里，每天的平均波能流起伏有多大，能不能预报。

与 Jin 2025 的 MVI（年内各月之间的起伏）不同，这里是起报后滚动窗口内各天之间的起伏，另起名字：
  原式：   (最大日平均波能流 − 最小日平均波能流) / 窗口平均波能流
  稳健版： (日平均波能流的 P90 − P10) / 窗口平均波能流（不受单个台风日左右）
  窗口：   第 1–30 天（第 16–30 天为气候态）、第 1–15 天（全部为模型），两个都报，不挑选。
波能流 P = 0.49·Hs²·Te（kW/m），逐 3 小时；日平均要求当天 8 个时刻至少 6 个有值；窗口内有效天数 ≥ 80% 才算这次起报，
预报、对照都只用实测有效的那些天（三者在同样的天上比较）。

预报（指数的概率分布）：
  - 每个时刻的边缘分布与 30 天产品相同：Hs = 模型 99 个分位数（第 14–15 天按 --blend-days 逐步过渡到气候态，产品默认 2），
    第 16–30 天 = 训练年浮标、同日期 ±15 天的气候态；Te = 周期多任务模型的分位数（有实测 Te 的关岛、夏威夷），
    马罗岛用 GEFS 原始 Te（与产品一致），第 16–30 天用训练年实测 Te 的同期气候态。
  - 时间顺序：Schaake 洗牌（Clark et al. 2004）。模板 = 训练年同一日历窗口（日期前后 ±15 天，每 3 天一个）的实测 Hs、Te 轨迹，
    随机取 N 条（固定种子）；每个时刻把 N 个边缘样本按模板在该时刻的秩次排好，Hs 与 Te 用同一模板 → 情景的逐日起伏和
    Hs–Te 搭配都来自真实历史。不能直接用 P50 / 均值序列算最大、最小（会系统偏小，表里"均值序列直接算"一行即示意）。
  - 每条情景各算一次指数 → N 个值。
对照（历史同期）：同样 N 条模板轨迹本身的指数（只用训练年实测）。
评分：指数的 CRPS（样本估计），技巧 = 1 − CRPS预报 / CRPS对照；P10–P90 覆盖；预报中位数与实测的秩相关；
按起报年月分块配对自助 2000 次（route_models.boot）。只读已保存的预测，不改模型，不参与任何挑选。
    python forecast_intramonth_variability.py                                    本机：只有 Transformer 的验证集预测
    python forecast_intramonth_variability.py --ensemble-with GKH_h15_gru_pretrain   服务器：最终集成
  确定"产品以稳健版第 1–15 天为主"后，定义冻结，再在测试集与业务数据上各描述性评一次（不再改定义）：
    python forecast_intramonth_variability.py --split test --test-dir server_upload/test_results_1028 --ensemble-with GKH_h15_gru_pretrain
    python forecast_intramonth_variability.py --split oper --ensemble-with GKH_h15_gru_pretrain     服务器（需要业务数据与 GPU）
输出：results/intramonth_variability_<配置>[_ens]_<split>.md 与 _cases.csv（每个站点起报 × 窗口一行）
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from export_quantile_forecasts import (FLUX, cfg_of, quantiles_of_config, site_climatology, observed, observed_te,  # noqa: E402
                                       te_quantile_rows)
from compare_blend_days import blend_weight                               # noqa: E402
from route_models import boot                                             # noqa: E402
from wavecast.data import feature_names                                   # noqa: E402
from wavecast.experiment import load_sites, build_all                     # noqa: E402
from wavecast.metrics import TAU                                          # noqa: E402

NAMES = {'G': '关岛', 'K': '马罗岛', 'H': '夏威夷'}
DAYS, STEP = 30, 3
PER_DAY = 24 // STEP
SPLITS = {'val': '验证集', 'test': '2017–2019 测试集', 'oper': '2020–2024 业务预报'}
SPAN = {'val': '验证期 G 2014–2016、K 2015–2016、H 2016–2017', 'test': '测试期 G、K 2017–2019，H 2018–2019',
        'oper': '业务预报 2020-09 起至浮标资料末（2024 年底），冻结模型、不调参'}
INDICES = {'orig': '原式 (最大 − 最小) / 平均', 'robust': '稳健 (P90 − P10) / 平均'}


def te_climatology(sites, window=15):
    """Per site and day of year: quantiles (at TAU) of observed Te in the training years, ±window days."""
    te_v = observed_te(sites)[0]
    out = {}
    for c, _, _ in sites:
        sid = str(c['site_id'])
        t = te_v[sid].dropna()
        t = t[t.index.year.isin(c['train_years'])]
        dy = t.index.dayofyear.to_numpy()
        out[sid] = {d: np.quantile(t[np.minimum(abs(dy - d), 366 - abs(dy - d)) <= window], TAU) for d in range(1, 367)}
    return out


def at_levels(q, n):
    """(T, 99) quantiles → (T, n) values at probabilities (i + 0.5) / n (linear between the 99 points, ends held)."""
    pos = np.clip((np.arange(n) + .5) / n * 99 - .5, 0, 98)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, 98)
    w = pos - lo
    return q[:, lo] * (1 - w) + q[:, hi] * w


def schaake(sorted_vals, template, rng):
    """sorted_vals (T, N) ascending per row; template (N, T). Returns (N, T): value of rank r goes to the member
    whose template is r-th smallest at that time (ties broken at random)."""
    jitter = rng.random(template.shape) * 1e-9
    ranks = np.argsort(np.argsort(template + jitter, axis=0), axis=0)
    return np.take_along_axis(sorted_vals.T, ranks, axis=0)


def index_of(daily, valid):
    """daily (..., D) daily-mean flux; valid (D,) bool. Returns dict of the two indices over the valid days."""
    d = daily[..., valid]
    m = d.mean(-1)
    return {'orig': (d.max(-1) - d.min(-1)) / m,
            'robust': (np.quantile(d, .9, axis=-1) - np.quantile(d, .1, axis=-1)) / m}


def crps_sample(x, y):
    """x (n, N) samples, y (n,) → CRPS per row = E|X − y| − ½ E|X − X'|."""
    x = np.sort(x, axis=1)
    n = x.shape[1]
    w = 2 * np.arange(1, n + 1) - n - 1
    return np.abs(x - y[:, None]).mean(1) - (x * w).sum(1) / n ** 2


def model_key(name):
    c = cfg_of(name)
    return c.get('models', ['transformer'])[0]


def forecast_rows(a):
    """(cfg, sites for observations / climatology, samples, input names, Hs quantiles, Te quantiles, label).
    Observations, climatology and templates always come from the original sites (training years, buoy record to 2024)."""
    label = a.config + ('_ens' if a.ensemble_with else '')
    if a.split == 'val':
        cfg, ablation, q = quantiles_of_config(a.config, 'val')
        if a.ensemble_with:
            q2 = quantiles_of_config(a.ensemble_with, 'val')[2]
            if q2.shape != q.shape:
                raise SystemExit('两个配置的预报行数不同，不能集成')
            q = (np.asarray(q) + np.asarray(q2)) / 2
        sites, _ = load_sites(cfg)
        samples, _, names = build_all(sites, cfg, feature_names(cfg, ablation))
        smp = samples['val']
        tq = te_quantile_rows(a.te_config, 'val', len(q))
        return cfg, sites, smp, names, np.asarray(q, float), tq, label
    if a.split == 'test':
        cfg = cfg_of(a.config)
        ablation = json.load(open(os.path.join(cfg['output_dir'], 'selection.json'), encoding='utf-8'))['best'][model_key(a.config)]['ablation']
        run_dir = lambda name: (os.path.join(a.test_dir, 'model_runs', name, 'test') if a.test_dir
                                else os.path.join(cfg_of(name)['output_dir'], 'test'))
        q = np.load(os.path.join(run_dir(a.config), 'quantiles.npz'))[model_key(a.config)]
        if a.ensemble_with:
            q = (q + np.load(os.path.join(run_dir(a.ensemble_with), 'quantiles.npz'))[model_key(a.ensemble_with)]) / 2
        tq = np.load(os.path.join(run_dir(a.te_config), 'te_quantiles.npz'))['quantiles']
        sites, _ = load_sites(cfg)
        samples, _, names = build_all(sites, cfg, feature_names(cfg, ablation))
        smp = samples['test']
        if not (len(q) == len(tq) == sum(len(s['y']) for s in smp)):
            raise SystemExit('测试集预测行数与样本不一致')
        return cfg, sites, smp, names, np.asarray(q, float), tq, label
    # oper: frozen models on the 2020–2024 GEFSv12 operational forecasts, exactly as evaluate_operational.py
    import torch
    from torch.utils.data import DataLoader
    from evaluate_operational import frozen, operational_cfg, predict_frozen
    from wavecast.data import Scaler
    from wavecast.experiment import make_model, predict_te, tensors
    years = list(range(a.years[0], a.years[1] + 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg, model, ablation = frozen(a.config)
    op = operational_cfg(cfg, a.suffix, years)
    sites_op, _ = load_sites(op)
    samples, skipped, names = build_all(sites_op, op, feature_names(cfg, ablation))
    smp = samples['test']
    if not smp:
        raise SystemExit(f'业务数据里没有可用起报（跳过原因 {skipped}）')
    print(f'业务数据：{len(smp)} 次站点起报；设备 {device}；冻结模型预测 …', flush=True)
    q = predict_frozen(cfg, model, ablation, smp, device)
    if a.ensemble_with:
        c2, m2, ab2 = frozen(a.ensemble_with)
        if feature_names(c2, ab2) != feature_names(cfg, ablation):
            raise SystemExit('两个运行的输入特征不同')
        q = (q + predict_frozen(c2, m2, ab2, smp, device)) / 2
    ct, mt, abt = frozen(a.te_config)
    if feature_names(ct, abt) != feature_names(cfg, ablation):
        raise SystemExit('周期模型的输入特征与波高模型不同，样本无法对齐')
    tes = []
    for seed in ct['seeds']:
        state = torch.load(os.path.join(ct['output_dir'], f'{mt}_{abt}_seed{seed}.pt'), map_location=device, weights_only=False)
        scaler = Scaler(); scaler.mean = np.array(state['scaler']['mean']); scaler.std = np.array(state['scaler']['std'])
        net = make_model(mt, state['n_features'], state['history'], state['horizon'], ct).to(device)
        net.load_state_dict(state['model'])
        tes.append(predict_te(net, DataLoader(tensors(smp, scaler), batch_size=ct['batch_size']), device))
    sites, _ = load_sites(cfg)
    return cfg, sites, smp, names, np.asarray(q, float), np.mean(tes, axis=0), label


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='GKH_h15_transformer_pretrain')
    p.add_argument('--ensemble-with', help='与另一个配置（GKH_h15_gru_pretrain）分位数平均：最终集成')
    p.add_argument('--te-config', default='GKH_h15_te_transformer_pretrain')
    p.add_argument('--split', choices=['val', 'test', 'oper'], default='val',
                   help='val：验证集；test：2017–2019 测试集（读已保存的冻结预测）；oper：2020–2024 业务预报（服务器，冻结模型现算）')
    p.add_argument('--test-dir', help='测试集结果目录（含 model_runs/<配置>/test/），如 server_upload/test_results_1028；'
                                      '不给则读各配置自己的 output_dir/test')
    p.add_argument('--suffix', default='_oper', help='oper：业务数据目录后缀（同 evaluate_operational.py）')
    p.add_argument('--years', nargs=2, type=int, default=[2020, 2025], help='oper：业务年份')
    p.add_argument('--blend-days', type=int, default=2, help='与产品一致（10-29 定为 2）')
    p.add_argument('--n', type=int, default=40, help='情景数 = 模板轨迹数')
    p.add_argument('--offset-days', type=int, default=15)
    p.add_argument('--offset-step', type=int, default=3)
    a = p.parse_args()

    cfg, sites, smp, names, q, tq, label = forecast_rows(a)
    meta = pd.DataFrame([m for s in smp for m in s['metadata']])
    if len(meta) != len(q):
        raise SystemExit('预报行数与样本不一致')
    sids = meta.site_id.astype(str).to_numpy()
    vt = pd.to_datetime(meta.valid_time, utc=True)
    print(f'{label} {a.split}：{len(meta.drop_duplicates(["site_id", "issue_time"]))} 个站点起报，读取气候态 …', flush=True)
    clim = site_climatology(sites)
    tclim = te_climatology(sites)
    w = blend_weight(meta.lead_hours.to_numpy(), cfg, a.blend_days)[:, None]
    hq = np.stack([clim[s][d]['hs'] for s, d in zip(sids, vt.dt.dayofyear)])
    q = (1 - w) * q + w * hq
    gefs_te = np.concatenate([s['future'][:, names.index('te')] for s in smp])
    has_spec = {str(c['site_id']) for c, _, _ in sites
                if os.path.exists(os.path.join(ROOT, 'processed', 'buoy_spectral')) and
                any(f.startswith(f"{c['site_id']}_") for f in os.listdir(os.path.join(ROOT, 'processed', 'buoy_spectral')))}
    tq = np.where(np.isin(sids, list(has_spec))[:, None], tq, gefs_te[:, None])

    hs_obs, te_obs = observed(sites, 'hs'), observed_te(sites)[0]
    grids = {}
    for c, _, _ in sites:
        sid = str(c['site_id'])
        idx = pd.date_range(hs_obs[sid].first_valid_index(), hs_obs[sid].last_valid_index(), freq=f'{STEP}h')
        grids[sid] = (idx[0], hs_obs[sid].reindex(idx).to_numpy(), te_obs[sid].reindex(idx).to_numpy(), set(c['train_years']))

    def window(sid, start):
        """Observed Hs, Te on the 240 steps after `start` (NaN outside the record)."""
        t0, hs, te, _ = grids[sid]
        k = int((start - t0) / pd.Timedelta(hours=STEP)) + np.arange(1, DAYS * PER_DAY + 1)
        ok = (k >= 0) & (k < len(hs))
        h, t = np.full(len(k), np.nan), np.full(len(k), np.nan)
        h[ok], t[ok] = hs[k[ok]], te[k[ok]]
        return h, t

    rng = np.random.default_rng(0)
    offsets = np.arange(-a.offset_days, a.offset_days + 1, a.offset_step)
    rows, skipped = [], 0
    meta['_row'] = np.arange(len(meta))
    for (sid, issue), g in meta.groupby(['site_id', 'issue_time'], sort=False):
        sid = str(sid)
        dec = pd.Timestamp(g.decision_time.iloc[0])
        dec = dec.tz_localize('UTC') if dec.tzinfo is None else dec.tz_convert('UTC')
        times = dec + pd.to_timedelta(np.arange(1, DAYS * PER_DAY + 1) * STEP, unit='h')
        # 实测（逐日有效性）
        h_o, t_o = window(sid, dec)
        e_o = (FLUX * h_o ** 2 * t_o).reshape(DAYS, PER_DAY)
        ok_day = (~np.isnan(e_o)).sum(1) >= 6
        cnt = (~np.isnan(e_o)).sum(1)
        daily_o = np.where(ok_day, np.nansum(e_o, 1) / np.maximum(cnt, 1), np.nan) if ok_day.any() else None
        # 模板：训练年同一日历窗口
        train = grids[sid][3]
        cands = []
        for y in sorted(train):
            for o in offsets:
                try:
                    st = dec.replace(year=y) + pd.Timedelta(days=int(o))
                except ValueError:
                    st = dec.replace(year=y, day=28) + pd.Timedelta(days=int(o))
                end = st + pd.Timedelta(days=DAYS)
                if st.year not in train or end.year not in train:
                    continue
                h, t = window(sid, st)
                if np.isnan(h).mean() > .1 or np.isnan(t).mean() > .1:
                    continue
                h = pd.Series(h).interpolate(limit_direction='both').to_numpy()
                t = pd.Series(t).interpolate(limit_direction='both').to_numpy()
                cands.append((h, t))
        if len(cands) < 20 or daily_o is None:
            skipped += 1
            continue
        pick = rng.choice(len(cands), size=min(a.n, len(cands)), replace=False)
        n = len(pick)
        th = np.stack([cands[i][0] for i in pick])
        tt = np.stack([cands[i][1] for i in pick])
        # 边缘分布：第 1–15 天模型（按 lead_hours 对齐），之后气候态
        hsq = np.empty((len(times), 99))
        teq = np.empty((len(times), 99))
        lead = ((times - dec) / pd.Timedelta(hours=1)).astype(int)
        pos = dict(zip(g.lead_hours.astype(int), g._row))
        doy = times.dayofyear
        for i, (L, d) in enumerate(zip(lead, doy)):
            if L in pos:
                hsq[i], teq[i] = q[pos[L]], tq[pos[L]]
            else:
                hsq[i], teq[i] = clim[sid][d]['hs'], tclim[sid][d]
        hs_s = schaake(np.sort(at_levels(hsq, n), axis=1), th, rng)
        te_s = schaake(np.sort(at_levels(teq, n), axis=1), tt, rng)
        e_f = (FLUX * hs_s ** 2 * te_s).reshape(n, DAYS, PER_DAY).mean(2)
        e_c = (FLUX * th ** 2 * tt).reshape(n, DAYS, PER_DAY).mean(2)
        e_mean = e_f.mean(0)                                         # 均值序列（与 P50 / 均值直接算的做法对照）
        for W in (30, 15):
            valid = ok_day.copy()
            valid[W:] = False
            if valid.sum() < .8 * W:
                continue
            io = index_of(daily_o, valid)
            ifc, icl, inv = index_of(e_f, valid), index_of(e_c, valid), index_of(e_mean, valid)
            for k in INDICES:
                r = dict(site_id=sid, issue_time=issue, window=W, index=k, days=int(valid.sum()), n=n,
                         observed=float(io[k]), naive=float(inv[k]))
                for name, x in (('fc', ifc[k]), ('cl', icl[k])):
                    r[f'crps_{name}'] = float(crps_sample(x[None], np.array([io[k]]))[0])
                    r[f'{name}_p10'], r[f'{name}_p50'], r[f'{name}_p90'] = np.quantile(x, [.1, .5, .9])
                rows.append(r)
    if not rows:
        raise SystemExit('没有可评的起报')
    d = pd.DataFrame(rows)
    d['issue_time'] = pd.to_datetime(d.issue_time, utc=True).dt.strftime('%Y-%m-%dT%H:%MZ')
    d['block'] = d.site_id + pd.to_datetime(d.issue_time, utc=True).dt.strftime('%Y-%m')

    lines = [f'# 月内日变化指数：能不能预报（{label}，{SPLITS[a.split]}）', '',
             '指数 = 窗口内日平均波能流的 (最大 − 最小) / 平均（原式），或 (P90 − P10) / 平均（稳健版）；窗口为起报后第 1–30 天'
             f'（第 16–30 天气候态）与第 1–15 天。产品衔接 --blend-days {a.blend_days}。情景最多 {a.n} 条（模板不足时用全部，至少 20 条）：边缘分布同 30 天产品，'
             '时间顺序由训练年同期实测轨迹 Schaake 洗牌给出；对照 = 同样的历史同期轨迹本身的指数。'
             f'跳过 {skipped} 个站点起报（同期模板 < 20 条或实测缺）。', '',
             '技巧 = 1 − CRPS预报 / CRPS对照（> 0 好于历史同期），括号为按站点 × 起报年月分块配对自助 2000 次的 95% 区间；'
             '覆盖 = 实测指数落在预报 P10–P90 内的比例（理想 80%）；秩相关 = 预报中位数与实测指数的 Spearman 相关。', '']
    for W in (30, 15):
        for k, kname in INDICES.items():
            lines += [f'## 第 1–{W} 天 · {kname}', '',
                      '| 站点 | 起报数 | 实测平均 | 预报中位数平均 | 均值序列直接算 | 历史同期中位数平均 | CRPS 预报 / 对照 | 技巧 | 覆盖 预报 / 对照 | 秩相关 预报 / 对照 |',
                      '|---|---:|---:|---:|---:|---:|---|---|---|---|']
            for sid in ['G', 'K', 'H', '三站']:
                g = d[(d.window == W) & (d['index'] == k) & ((d.site_id == sid) if sid != '三站' else True)]
                if g.empty:
                    continue
                ch, lo, hi = boot(g.crps_cl.to_numpy(), g.crps_fc.to_numpy(), g.block.to_numpy(), np.random.default_rng(0))
                cov = lambda pre: ((g.observed >= g[f'{pre}_p10']) & (g.observed <= g[f'{pre}_p90'])).mean()
                rf = spearmanr(g.fc_p50, g.observed).statistic
                rc = spearmanr(g.cl_p50, g.observed).statistic
                lines.append(f'| {NAMES.get(sid, sid)} | {len(g)} | {g.observed.mean():.2f} | {g.fc_p50.mean():.2f} | {g.naive.mean():.2f} | '
                             f'{g.cl_p50.mean():.2f} | {g.crps_fc.mean():.3f} / {g.crps_cl.mean():.3f} | '
                             f'{-ch:+.2f} [{-hi:+.2f}, {-lo:+.2f}] | {cov("fc"):.0%} / {cov("cl"):.0%} | {rf:.2f} / {rc:.2f} |')
            lines.append('')
    lines += [f'说明：{SPAN[a.split]}；相邻两周起报的 30 天窗口重叠，样本不独立（已按站点 × 年月分块）。'
              '马罗岛第 1–15 天的 Te 是 GEFS 原始单值（与产品一致），其"实测"波能流用 ERA5 Te；关岛、夏威夷的"实测"Te 用 CDIP 谱，'
              '谱缺测的时刻用 ERA5 Te（2023–2024 夏威夷、2024 关岛多为 ERA5）。']
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    base = os.path.join(ROOT, 'results', f'intramonth_variability_{label}_{a.split}')
    open(base + '.md', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    d.round(4).to_csv(base + '_cases.csv', index=False, encoding='utf-8-sig')
    print('\n'.join(lines))
    print(f'→ {base}.md')


if __name__ == '__main__':
    main()
