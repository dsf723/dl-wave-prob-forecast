"""第 15 → 16 天模型与气候态的衔接：比较 export_quantile_forecasts.py --blend-days N 的几个取值（10-29，只用验证集）。

    python compare_blend_days.py --config GKH_h15_transformer_pretrain --ensemble-with GKH_h15_gru_pretrain
    python compare_blend_days.py --config GKH_h15_transformer_pretrain            # 只有 Transformer 时（本机）

做法与 export_quantile_forecasts.py 完全相同：模型最后 N 天的分位数与气候态（训练年浮标、预见日期 ±15 天）按权重线性平均，
权重从 0 逐步升到 N/(N+1)；第 16 天起都是气候态，所以只有第 1–15 天的 CRPS 会变。
标准（事先定）：看第 12–15 天（--late-from）的验证集 CRPS 会不会变差：相对 N = 0 按起报块配对自助 95% 区间
下界 > 0 算显著变差；不变差的 N 里取这一段 CRPS 最低的。第 1–15 天整体 CRPS 一并报告。另报衔接处跳变：最后一个模型时刻与第一个气候态时刻的 P50 差。
输出：results/blend_days_<配置>.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from export_quantile_forecasts import quantiles_of_config, site_climatology  # noqa: E402
from wavecast.data import feature_names                                     # noqa: E402
from wavecast.experiment import load_sites, build_all                       # noqa: E402
from wavecast.metrics import score, quantile_at, paired_bootstrap          # noqa: E402


def blend_weight(lead_hours, cfg, n):
    """同 export_quantile_forecasts.main 的 --blend-days。"""
    if not n:
        return np.zeros(len(lead_hours))
    steps = n * 24 // cfg['step_hours']
    horizon = cfg['horizon_days'] * 24 // cfg['step_hours']
    j = np.asarray(lead_hours) // cfg['step_hours']
    return np.clip((j - (horizon - steps)) / (steps + 1), 0, 1)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='GKH_h15_transformer_pretrain')
    p.add_argument('--ensemble-with')
    p.add_argument('--blend', type=int, nargs='+', default=[0, 2, 3, 5])
    p.add_argument('--late-from', type=int, default=12, help='衔接段从第几天起单独比较（默认第 12–15 天）')
    a = p.parse_args()
    cfg, ablation, q = quantiles_of_config(a.config, 'val')
    label = a.config
    if a.ensemble_with:
        _, _, q2 = quantiles_of_config(a.ensemble_with, 'val')
        q, label = (np.asarray(q) + np.asarray(q2)) / 2, label + '_ens'
    sites, _ = load_sites(cfg)
    samples, _, _ = build_all(sites, cfg, feature_names(cfg, ablation))
    smp = samples['val']
    meta = pd.DataFrame([m for s in smp for m in s['metadata']])
    y = np.concatenate([s['y'] for s in smp])
    if len(q) != len(y):
        raise SystemExit('预报行数与样本不一致')
    clim = site_climatology(sites)
    doy = pd.to_datetime(meta.valid_time, utc=True).dt.dayofyear.to_numpy()
    hq = np.stack([clim[str(s)][d]['hs'] for s, d in zip(meta.site_id, doy)])
    ok = np.isfinite(y)
    meta, y, q, hq = meta[ok].reset_index(drop=True), y[ok], np.asarray(q)[ok], hq[ok]
    print(f'{label}：验证集 {len(y):,} 个有实测的时刻，{meta[["site_id", "issue_time"]].drop_duplicates().shape[0]} 次站点起报', flush=True)

    # 衔接处：每次起报最后一个模型时刻；第一个气候态时刻的 P50（预见日期 +3 小时）
    last = meta.groupby(['site_id', 'issue_time']).lead_hours.transform('max').eq(meta.lead_hours).to_numpy()
    nxt = (pd.to_datetime(meta.valid_time, utc=True) + pd.Timedelta(hours=cfg['step_hours'])).dt.dayofyear.to_numpy()
    clim_next50 = np.array([np.interp(.5, (np.arange(99) + .5) / 99, clim[str(s)][d]['hs']) for s, d in zip(meta.site_id, nxt)])

    tables, rows = {}, []
    for n in a.blend:
        w = blend_weight(meta.lead_hours, cfg, n)[:, None]
        qn = (1 - w) * q + w * hq
        t = meta[['site_id', 'issue_time', 'valid_time', 'block_id', 'lead_day']].copy()
        t['crps'] = score(qn, y)
        p10, p50, p90 = (quantile_at(qn, v) for v in (.1, .5, .9))
        t['cov'] = (y >= p10) & (y <= p90)
        tables[n] = t
        jump = np.abs(p50[last] - clim_next50[last])
        rec = dict(N=n, crps=t.crps.mean(), late=t.crps[t.lead_day >= a.late_from].mean(), cov_late=t['cov'][t.lead_day >= a.late_from].mean(),
                   jump_mean=jump.mean(), jump_p95=np.quantile(jump, .95))
        rec.update({f'crps_{s}': g.crps.mean() for s, g in t.groupby(t.site_id.astype(str))})
        rows.append(rec)
        print(f'  N = {n}：CRPS {rec["crps"]:.4f}，第 {a.late_from}–15 天 {rec["late"]:.4f}，衔接跳变平均 {rec["jump_mean"]:.3f} m', flush=True)
    res = pd.DataFrame(rows)
    base = tables[a.blend[0]] if a.blend[0] == 0 else tables[0]
    for i, n in enumerate(res.N):
        b = paired_bootstrap(tables[n], base, repeats=2000)
        late = paired_bootstrap(tables[n][tables[n].lead_day >= a.late_from], base[base.lead_day >= a.late_from], repeats=2000)
        m = base.crps.mean()
        ml = base.crps[base.lead_day >= a.late_from].mean()
        res.loc[i, 'rel'] = b['delta'] / m
        res.loc[i, 'lo'], res.loc[i, 'hi'] = (b['ci_low'] or 0) / m, (b['ci_high'] or 0) / m
        res.loc[i, 'rel_late'] = late['delta'] / ml
        res.loc[i, 'lo_late'], res.loc[i, 'hi_late'] = (late['ci_low'] or 0) / ml, (late['ci_high'] or 0) / ml
    # 事先定的标准：第 late_from–15 天 CRPS 不显著变差（95% 区间下界 > 0 才算显著变差）；不变差的 N 里取这一段 CRPS 最低的
    ok_n = res[res.lo_late <= 0]
    pick = int(ok_n.sort_values('late').N.iloc[0])

    lines = [f'# 第 15 → 16 天衔接：--blend-days 比较（验证集，{label}）', '',
             f'验证集 {len(y):,} 个有实测的时刻；只有第 1–15 天的预报会变（第 16 天起都是气候态）。'
             '变化 = 相对 N = 0（直接切换），按起报块配对自助 2000 次的 95% 区间。', '',
             f'| N（天） | CRPS 第 1–15 天 | 相对 N = 0 | 第 {a.late_from}–15 天 CRPS | 相对 N = 0 | 第 {a.late_from}–15 天覆盖 | 衔接跳变 P50 平均 / 95% (m) |'
             + ''.join(f' {s} |' for s in sorted(c for c in res if c.startswith('crps_'))),
             '|---:|---:|---|---:|---|---:|---|' + '---:|' * sum(c.startswith('crps_') for c in res)]
    for _, r in res.iterrows():
        lines.append(f'| {int(r.N)} | {r.crps:.4f} | {r.rel:+.2%} [{r.lo:+.2%}, {r.hi:+.2%}] | {r.late:.4f} | '
                     f'{r.rel_late:+.2%} [{r.lo_late:+.2%}, {r.hi_late:+.2%}] | {r.cov_late:.1%} | {r.jump_mean:.3f} / {r.jump_p95:.3f} |'
                     + ''.join(f' {r[c]:.4f} |' for c in sorted(c for c in res if c.startswith('crps_'))))
    worse = [int(v) for v in res[res.lo_late > 0].N]
    lines += ['', f'标准（10-29 定）：第 {a.late_from}–15 天 CRPS 不显著变差；其中这一段 CRPS 最低的 → **N = {pick}**'
              + (f'；显著变差：N = {worse}' if worse else '；没有显著变差的 N')]
    path = os.path.join(ROOT, 'results', f'blend_days_{label}.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'→ {path}')


if __name__ == '__main__':
    main()
