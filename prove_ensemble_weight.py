"""用多种独立方法检验集成权重 w = 0.5（只用验证集，不碰测试集）。
q = w·q_Transformer + (1 − w)·q_GRU，总分 S = CRPS + twCRPS(2.5 m)。
  1. 最小 CRPS 估计（Gneiting et al. 2005）：连续优化，并核对 S(w) 的凸性。
  2. Bates & Granger (1969) 最优组合权重：由两模型中位数误差的方差、协方差直接算出的闭式解。
  3. 等效性检验：块自助 2000 次，每次重新找最优 w，遗憾 = S(0.5) ÷ min S(w) − 1；95 分位 ≤ 0.5% 判等效（标准事先定）。
  4. 样本外检验：留一站、留一年、5 折分块；单一 w、分时效、分站、分分位数段的权重与固定 0.5 比较。
  5. 独立训练检验：用一组种子选 w，在另一组种子训练的模型上评分。
用法：python prove_ensemble_weight.py --t-dir <Transformer 运行目录> --g-dir <GRU 运行目录>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from route_models import boot                                                # noqa: E402
from wavecast.metrics import TAU, score                                      # noqa: E402

MARGIN = .005
WS = np.round(np.arange(0, 1 + 1e-9, .05), 2)
OUT = os.path.join(ROOT, 'results', 'weights')


def seed_q(run_dir, model, seeds=(11, 23, 37)):
    return np.mean([np.load(os.path.join(run_dir, f'{model}_season_seed{s}_val.npz'))['quantiles'] for s in seeds], axis=0)


def tw(q, y):
    return score(np.maximum(q, 2.5), np.maximum(y, 2.5))


def total(q, y):
    return score(q, y) + tw(q, y)


def per_tau(q, y):
    """Total score per row and quantile level (rows × 99); its mean over the levels is total(q, y)."""
    def pin(q, y):
        e = y[:, None] - q
        return 2 * np.maximum(TAU * e, (TAU - 1) * e)
    return (pin(q, y) + pin(np.maximum(q, 2.5), np.maximum(y, 2.5))).astype(np.float32)


def verdict(lo, hi):
    return '显著变好' if hi < 0 else ('显著变差' if lo > 0 else '不显著')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--t-dir', required=True)
    p.add_argument('--g-dir', required=True)
    a = p.parse_args()
    cs = pd.read_csv(os.path.join(ROOT, 'model_runs', 'GKH_h15_solo_base_transformer_pretrain', 'validation', 'case_scores.csv'),
                     usecols=['model', 'truth', 'block_id', 'site_id', 'issue_time', 'lead_day'])
    cs = cs[cs.model == 'transformer_season'].reset_index(drop=True)
    y, blocks = cs.truth.to_numpy(float), cs.block_id.to_numpy()
    qt, qg = seed_q(a.t_dir, 'transformer'), seed_q(a.g_dir, 'gru')
    rng = np.random.default_rng(0)
    half = (qt + qg) / 2
    L = ['# 集成权重 w 的检验（验证集）', '',
         f'Transformer：`{os.path.basename(a.t_dir)}`，GRU：`{os.path.basename(a.g_dir)}`（各 3 个种子平均）；{len(y)} 个时刻。总分 S = CRPS + twCRPS(2.5 m)。', '',
         f'S：Transformer 单独 {total(qt, y).mean():.4f}，GRU 单独 {total(qg, y).mean():.4f}，w = 0.5 集成 {total(half, y).mean():.4f}'
         f'（CRPS {score(half, y).mean():.4f}，twCRPS {tw(half, y).mean():.4f}）。', '']
    for name, q in (('Transformer 单独', qt), ('GRU 单独', qg)):
        d, lo, hi = boot(total(half, y), total(q, y), blocks, rng)
        L.append(f'- {name}相对 w = 0.5 集成：{d:+.2%} [{lo:+.2%}, {hi:+.2%}]，{verdict(lo, hi)}。')

    # 1. minimum-score estimation and convexity
    f = lambda w: float(total(w * qt + (1 - w) * qg, y).mean())
    r = minimize_scalar(f, bounds=(0, 1), method='bounded', options=dict(xatol=1e-3))
    fine = np.linspace(0, 1, 41)
    v = np.array([f(w) for w in fine])
    cv = np.array([float(score(w * qt + (1 - w) * qg, y).mean()) for w in fine])
    L += ['', '## 1. 最小 CRPS 估计与凸性', '',
          f'- 连续优化得 w* = {r.x:.3f}，S = {r.fun:.5f}；w = 0.5 时 S = {f(.5):.5f}，相对 {f(.5) / r.fun - 1:+.3%}。',
          f'- 分位数损失对分位数值是凸函数，所以 CRPS 对 w 必为凸函数，不会有多个分离的极小值（数学性质）。数值核对：CRPS 二阶差分最小 {np.diff(cv, 2).min():+.1e}，'
          f'总分 S 二阶差分最小 {np.diff(v, 2).min():+.1e}（twCRPS 项没有这个数学保证，只作数值核对）。',
          f'- 由凸性，任何 w 的 CRPS 都不超过两个单模型的加权平均：w = 0.5 时 {score(half, y).mean():.4f} ≤ {(score(qt, y).mean() + score(qg, y).mean()) / 2:.4f}。']

    # 2. Bates & Granger
    def bg(i):
        et, eg = qt[:, 49][i] - y[i], qg[:, 49][i] - y[i]
        c = np.cov(et, eg)
        return (c[1, 1] - c[0, 1]) / (c[0, 0] + c[1, 1] - 2 * c[0, 1]), c
    w_bg, c = bg(np.arange(len(y)))
    rho = c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
    ub = np.unique(blocks)
    rows_of = {b: np.flatnonzero(blocks == b) for b in ub}
    wb = [bg(np.concatenate([rows_of[b] for b in rng.choice(ub, len(ub))]))[0] for _ in range(2000)]
    L += ['', '## 2. Bates & Granger 闭式解（中位数误差）', '',
          f'- 误差标准差：Transformer {c[0, 0] ** .5:.4f} m，GRU {c[1, 1] ** .5:.4f} m，相关系数 {rho:.3f}。',
          f'- w = (σ_G² − ρσ_Tσ_G) / (σ_T² + σ_G² − 2ρσ_Tσ_G) = {w_bg:.3f}，块自助 95% 区间 [{np.percentile(wb, 2.5):.3f}, {np.percentile(wb, 97.5):.3f}]。',
          f'- 忽略相关（Bates & Granger 建议的简化，按均方误差倒数加权）：w = {np.mean((qg[:, 49] - y) ** 2) / (np.mean((qt[:, 49] - y) ** 2) + np.mean((qg[:, 49] - y) ** 2)):.3f}。']

    # 3. equivalence through regret, the optimum re-estimated in every resample
    LT = np.stack([per_tau(w * qt + (1 - w) * qg, y) for w in WS])    # weights × rows × quantile levels
    S = LT.mean(2).T                                                  # rows × weights
    i5 = int(np.where(WS == .5)[0][0])
    bs = pd.DataFrame(S).groupby(blocks).sum().to_numpy()
    idx = rng.integers(0, len(bs), (2000, len(bs)))
    res = bs[idx].sum(1)
    regret = res[:, i5] / res.min(1) - 1
    w_star = WS[res.argmin(1)]
    q95 = float(np.percentile(regret, 95))
    L += ['', '## 3. 等效性检验（事先定的标准：遗憾 95 分位 ≤ 0.5%）', '',
          f'- 每次重抽样重新找最优 w：最优 w 中位数 {np.median(w_star):.2f}，95% 区间 [{np.percentile(w_star, 2.5):.2f}, {np.percentile(w_star, 97.5):.2f}]。',
          f'- w = 0.5 比当次最优差：中位数 {np.median(regret):.3%}，95 分位 {q95:.3%}，最大 {regret.max():.3%}。',
          f'- 当次最优落在 0.45–0.55 的比例 {np.mean((w_star >= .45) & (w_star <= .55)):.1%}。',
          f'- **判定：{"等效" if q95 <= MARGIN else "未达到等效标准"}**（{q95:.3%} {"≤" if q95 <= MARGIN else ">"} {MARGIN:.1%}）。']

    # 4. out-of-sample schemes
    site, lead = cs.site_id.to_numpy(), np.where(cs.lead_day <= 3, 'D1-3', np.where(cs.lead_day <= 7, 'D4-7', 'D8-15'))
    qgroup = np.where(np.arange(99) < 25, 0, np.where(np.arange(99) < 75, 1, 2))
    years = cs.issue_time.str[:4].to_numpy()
    key = pd.Series(site + '_' + cs.issue_time.str[:7].to_numpy())
    uk = np.array(sorted(key.unique()))
    fold5 = key.map(dict(zip(uk, rng.permutation(len(uk)) % 5))).to_numpy()

    def oos(folds, scheme):
        out = np.zeros(len(y))
        chosen = []
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            if scheme == 'single':
                j = int(LT[:, tr].mean((1, 2)).argmin())
                out[te] = LT[j, te].mean(1)
                chosen.append(float(WS[j]))
            elif scheme in ('lead', 'site'):
                g = lead if scheme == 'lead' else site
                for b in np.unique(g):
                    trb, teb = tr & (g == b), te & (g == b)
                    if teb.any():
                        j = int(LT[:, trb if trb.any() else tr].mean((1, 2)).argmin())
                        out[teb] = LT[j, teb].mean(1)
            else:                                                     # one weight per quantile segment
                parts = np.zeros((int(te.sum()), 99), dtype=np.float32)
                for b in range(3):
                    m = qgroup == b
                    j = int(LT[:, tr][:, :, m].mean((1, 2)).argmin())
                    parts[:, m] = LT[j, te][:, m]
                out[te] = parts.mean(1)
        return out, chosen
    L += ['', '## 4. 样本外检验（在留出数据上相对固定 w = 0.5）', '', '| 留出方式 | 权重方案 | 相对 0.5 | 判断 | 各折选出的 w |', '|---|---|---|---|---|']
    for fname, folds, schemes in (('留一站', site, ('single', 'lead', 'quantile')), ('留一年', years, ('single', 'lead', 'site', 'quantile')),
                                  ('5 折分块', fold5, ('single', 'lead', 'site', 'quantile'))):
        for sch in schemes:
            s, ch = oos(folds, sch)
            d, lo, hi = boot(S[:, i5], s, blocks, rng)
            lab = dict(single='单一 w', lead='分时效', site='分站', quantile='分分位数段')[sch]
            L.append(f'| {fname} | {lab} | {d:+.3%} [{lo:+.3%}, {hi:+.3%}] | {verdict(lo, hi)} | {", ".join(f"{x:.2f}" for x in ch)} |')
    L += ['', '各子集各自的最优 w（只供参考）：' + '；'.join(f'{b} {WS[int(S[site == b].mean(0).argmin())]:.2f}' for b in np.unique(site)) + '；'
          + '；'.join(f'{b} {WS[int(S[lead == b].mean(0).argmin())]:.2f}' for b in ('D1-3', 'D4-7', 'D8-15')) + '。']

    # 5. weights chosen on independently trained seeds
    run = lambda tag, m: os.path.join(ROOT, 'model_runs', tag.format(m=m))
    A = ('GKH_h15_solo_base_{m}_pretrain', (11, 23, 37))
    B = ('GKH_h15_why6_base_s2_{m}_pretrain', (41, 53, 67))
    L += ['', '## 5. 独立训练检验（本机：一组种子选 w，另一组种子上评分）', '', '| 选 w 用的种子 | 选出的 w | 评分用的种子 | 选出的 w 相对 0.5 | 判断 |', '|---|---:|---|---|---|']
    for (fit, sf), (ev, se) in ((A, B), (B, A)):
        qa, qb = seed_q(run(fit, 'transformer'), 'transformer', sf), seed_q(run(fit, 'gru'), 'gru', sf)
        w = WS[int(np.argmin([total(x * qa + (1 - x) * qb, y).mean() for x in WS]))]
        qc, qd = seed_q(run(ev, 'transformer'), 'transformer', se), seed_q(run(ev, 'gru'), 'gru', se)
        d, lo, hi = boot(total((qc + qd) / 2, y), total(w * qc + (1 - w) * qd, y), blocks, rng)
        L.append(f'| {"、".join(map(str, sf))} | {w:.2f} | {"、".join(map(str, se))} | {d:+.3%} [{lo:+.3%}, {hi:+.3%}] | {verdict(lo, hi)} |')

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, MUTED = '#22221f', '#6f6e68'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=200)
    lo95, hi95 = np.percentile(w_star, [2.5, 97.5])
    ax.axvspan(lo95, hi95, color='#b7d3f6', alpha=.6, lw=0, label=f'95% range of best w ({lo95:.2f}–{hi95:.2f})')
    ax.plot(fine, v, color='#2a78d6', lw=1.5, label='Validation score')
    ax.scatter([.5], [f(.5)], s=48, color='#eb6834', edgecolor='white', linewidth=.8, zorder=4, label=f'w = 0.5 ({f(.5):.4f})')
    ax.scatter([r.x], [r.fun], s=26, color=INK, zorder=5, label=f'Minimum, w = {r.x:.2f} ({r.fun:.4f})')
    ax.set_xlabel('Transformer weight w'); ax.set_ylabel('Validation CRPS + twCRPS (m)')
    ax.grid(color='#e8e7e2', lw=.6); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'ensemble_weight_proof.png')); plt.close(fig)
    out = os.path.join(OUT, 'ensemble_weight_proof.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L)); print(f'\n→ {out}')


if __name__ == '__main__':
    main()
