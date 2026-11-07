"""--blend-days 2 在测试集与独立业务检验上各评一次分（10-29）。

N = 2 已在验证集上定好（compare_blend_days.py；第 12–15 天 CRPS 不变差、最低），这里只是把定好的产品衔接用到
2017–2019 测试集和 2020–2024 业务数据上各评一次，与直接切换（N = 0，已报告的数字）并列；无论结果如何都不再改 N。
读已保存的预测，不重新预测：
  测试集：<test_dir>/model_runs/{A,B}/test/quantiles.npz、<test_dir>/results/postproc/<A>_test_emos.npz（EMOS 对数正态）
  业务：  <oper_dir>/quantiles.npz、cases.csv（evaluate_operational.py 的输出）
统计与 postproc_diagnostics / evaluate_operational 相同：按 block_id 配对自助 2000 次（route_models.boot）。
    python score_blend_test_oper.py server_upload/test_results_1028 \
        server_upload/operational_results_1028/results/operational/GKH_h15_transformer_pretrain_ens_oper
输出：results/blend2_test_oper.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from export_quantile_forecasts import cfg_of, site_climatology            # noqa: E402
from compare_blend_days import blend_weight                               # noqa: E402
from route_models import boot                                             # noqa: E402
from wavecast.data import feature_names                                   # noqa: E402
from wavecast.experiment import load_sites, build_all, climatology        # noqa: E402
from wavecast.metrics import score                                        # noqa: E402

A, B, N = 'GKH_h15_transformer_pretrain', 'GKH_h15_gru_pretrain', 2


def tw(q, y, t=2.5):
    return score(np.maximum(q, t), np.maximum(y, t))


def blended(q, site_ids, valid_times, lead_hours, cfg, clim, n):
    w = blend_weight(lead_hours, cfg, n)[:, None]
    doy = pd.to_datetime(valid_times, utc=True).dt.dayofyear.to_numpy()
    hq = np.stack([clim[str(s)][d]['hs'] for s, d in zip(site_ids, doy)])
    return (1 - w) * q + w * hq


def compare(name, q0, q2, refs, y, meta, lines):
    rng = np.random.default_rng(0)
    blocks = meta.block_id.to_numpy()
    late = (meta.lead_day >= 12).to_numpy()
    cov = lambda q, k: ((y[k] >= q[k, 9]) & (y[k] <= q[k, 89])).mean()
    lines += [f'## {name}（{len(y):,} 个有实测的时刻）', '',
              '| 衔接 | CRPS | 第 12–15 天 CRPS | 第 12–15 天覆盖 | ' + ' | '.join(f'相对{r}' for r in refs) + ' | twCRPS 相对 EMOS 对数正态 |',
              '|---|---:|---:|---:|' + '---|' * (len(refs) + 1)]
    for lab, q in (('直接切换 N = 0（已报告）', q0), (f'逐步过渡 N = {N}（产品）', q2)):
        cells = []
        for r, qr in refs.items():
            d, lo, hi = boot(score(qr, y), score(q, y), blocks, rng)
            cells.append(f'{d:+.1%} [{lo:+.1%}, {hi:+.1%}]')
        d, lo, hi = boot(tw(refs['EMOS 对数正态'], y), tw(q, y), blocks, rng)
        cells.append(f'{d:+.1%} [{lo:+.1%}, {hi:+.1%}]')
        lines.append(f'| {lab} | {score(q, y).mean():.4f} | {score(q[late], y[late]).mean():.4f} | {cov(q, late):.1%} | ' + ' | '.join(cells) + ' |')
    d, lo, hi = boot(score(q0, y), score(q2, y), blocks, rng)
    dl, lol, hil = boot(score(q0[late], y[late]), score(q2[late], y[late]), blocks[late], rng)
    lines += ['', f'N = {N} 相对 N = 0：全部 {d:+.2%} [{lo:+.2%}, {hi:+.2%}]；第 12–15 天 {dl:+.2%} [{lol:+.2%}, {hil:+.2%}]'
              f'（第 1–13 天两者完全相同）。', '']
    return d, dl


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('test_dir')
    p.add_argument('oper_dir')
    a = p.parse_args()
    cfg = cfg_of(A)
    sites, _ = load_sites(cfg)
    clim = site_climatology(sites)
    lines = [f'# 产品衔接 --blend-days {N}：测试集与独立业务检验各评一次', '',
             f'N = {N} 是 10-29 在验证集上定的（compare_blend_days.py），这里不再挑选，只与已报告的 N = 0 并列。'
             '变化为集成相对对照的 CRPS 变化，括号为按 block_id 配对自助 2000 次的 95% 区间。', '']

    # 2017–2019 test set
    import json
    abl = json.load(open(os.path.join(cfg['output_dir'], 'selection.json'), encoding='utf-8'))['best']['transformer']['ablation']
    samples, _, _ = build_all(sites, cfg, feature_names(cfg, abl))
    test = samples['test']
    meta = pd.DataFrame([m for s in test for m in s['metadata']])
    y = np.concatenate([s['y'] for s in test]).astype(float)
    qa = np.load(os.path.join(a.test_dir, 'model_runs', A, 'test', 'quantiles.npz'))['transformer']
    qb = np.load(os.path.join(a.test_dir, 'model_runs', B, 'test', 'quantiles.npz'))['gru']
    q0 = (qa + qb) / 2
    refs = {'GEFS 原始集合': np.concatenate([s['raw_quantiles'] for s in test]).astype(float),
            'EMOS 对数正态': np.load(os.path.join(a.test_dir, 'results', 'postproc', f'{A}_test_emos.npz'))['ln'],
            '气候态': climatology(sites, test, cfg['target'])}
    if not (len(q0) == len(y) == len(refs['EMOS 对数正态'])):
        raise SystemExit('测试集预测行数与样本不一致')
    q2 = blended(q0, meta.site_id, meta.valid_time, meta.lead_hours, cfg, clim, N)
    compare('2017–2019 回算测试集', q0, q2, refs, y, meta, lines)

    # 2020–2024 operational
    z = np.load(os.path.join(a.oper_dir, 'quantiles.npz'))
    cases = pd.read_csv(os.path.join(a.oper_dir, 'cases.csv'))
    y = cases.truth.to_numpy(float)
    lead_hours = ((pd.to_datetime(cases.valid_time, utc=True) - pd.to_datetime(cases.issue_time, utc=True)).dt.total_seconds() // 3600
                  - cfg['decision_delay_hours']).astype(int).to_numpy()
    q0 = z['ensemble']
    refs = {'GEFS 原始集合': z['raw_ensemble'], 'EMOS 对数正态': z['emos_ln'], '气候态': z['climatology']}
    q2 = blended(q0, cases.site_id, cases.valid_time, lead_hours, cfg, clim, N)
    compare('2020–2024 独立业务检验（主结果）', q0, q2, refs, y, cases, lines)

    path = os.path.join(ROOT, 'results', f'blend{N}_test_oper.md')
    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'→ {path}')


if __name__ == '__main__':
    main()
