"""用 6 个种子重判损失权重 λ = 1（10-31；只用验证集，不碰测试集）。λ ∈ {0, 0.5, 1, 2, 4}，种子 A = 11、23、37，B = 41、53、67。
  1. 遗憾：块自助 2000 次，每次在 5 个 λ 里重选最优，遗憾 = S(1) ÷ min S(λ) − 1；95 分位 ≤ 0.5% 判等效（标准事先定）。
     同时给出只用 3 个种子时的遗憾，区分"点集变小"与"种子变多"两种效果。
  2. 交叉验证：一组种子选 λ，另一组种子上与 λ = 1 配对比较。
  3. 方差分析：每个 λ 的 6 个单种子总分，λ 的影响是否超过种子噪声；并检验总分对 λ 的线性趋势。
  4. 集成：5 × 5 组 (λ_T, λ_G)、w = 0.5，同样算遗憾和交叉验证。
用法：python prove_lambda6.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from route_models import boot                                                # noqa: E402
from wavecast.metrics import score                                           # noqa: E402

MARGIN = .005
LAMS = [(0., '0'), (.5, '05'), (1., '1'), (2., '2'), (4., '4')]
SEEDS = {'A': (11, 23, 37), 'B': (41, 53, 67)}
OUT = os.path.join(ROOT, 'results', 'weights')


def run_dir(tag, m, group):
    if group == 'A':
        name = f'GKH_h15_lam{tag}_{m}_pretrain'
    elif tag == '1':
        name = f'GKH_h15_why6_base_s2_{m}_pretrain'
    else:
        name = f'GKH_h15_lam{tag}_s2_{m}_pretrain'
    return os.path.join(ROOT, 'model_runs', name)


def seed_quantiles(tag, m, group):
    return [np.load(os.path.join(run_dir(tag, m, group), f'{m}_season_seed{s}_val.npz'))['quantiles'] for s in SEEDS[group]]


def total(q, y):
    return score(q, y) + score(np.maximum(q, 2.5), np.maximum(y, 2.5))


def verdict(lo, hi):
    return '显著变好' if hi < 0 else ('显著变差' if lo > 0 else '不显著')


def regret(S, i1, blocks, rng):
    """S: rows × candidates. Regret of candidate i1 against the best candidate of each block-bootstrap resample."""
    bs = pd.DataFrame(S).groupby(blocks).sum().to_numpy()
    res = bs[rng.integers(0, len(bs), (2000, len(bs)))].sum(1)
    return res[:, i1] / res.min(1) - 1, res.argmin(1)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cs = pd.read_csv(os.path.join(ROOT, 'model_runs', 'GKH_h15_solo_base_transformer_pretrain', 'validation', 'case_scores.csv'),
                     usecols=['model', 'truth', 'block_id'])
    cs = cs[cs.model == 'transformer_season'].reset_index(drop=True)
    y, blocks = cs.truth.to_numpy(float), cs.block_id.to_numpy()
    rng = np.random.default_rng(0)
    lams = [x for x, _ in LAMS]
    i1 = lams.index(1.)
    L = ['# 用 6 个种子重判 λ（验证集）', '', f'λ ∈ {{0, 0.5, 1, 2, 4}}，种子 11、23、37 + 41、53、67；{len(y)} 个时刻；本机模型。总分 S = CRPS + twCRPS(2.5 m)。', '']
    Q = {}
    for m in ['transformer', 'gru']:
        by_seed = {g: {lam: seed_quantiles(tag, m, g) for lam, tag in LAMS} for g in SEEDS}
        q3 = {g: {lam: np.mean(v, axis=0) for lam, v in by_seed[g].items()} for g in SEEDS}
        q6 = {lam: (q3['A'][lam] + q3['B'][lam]) / 2 for lam in lams}
        Q[m] = dict(q3=q3, q6=q6)
        S6 = np.stack([total(q6[lam], y) for lam in lams], axis=1)
        S3 = {g: np.stack([total(q3[g][lam], y) for lam in lams], axis=1) for g in SEEDS}
        single = {lam: [float(total(q, y).mean()) for g in SEEDS for q in by_seed[g][lam]] for lam in lams}
        L += [f'## {m}', '', '| λ | 6 个种子平均的 S | 相对 λ = 1 | 判断 | 种子 A | 种子 B | 单种子 S 的标准差 |', '|---:|---:|---|---|---:|---:|---:|']
        for j, lam in enumerate(lams):
            d, lo, hi = boot(S6[:, i1], S6[:, j], blocks, rng)
            L.append(f'| {lam:g} | {S6[:, j].mean():.4f} | {"基准" if j == i1 else f"{d:+.2%} [{lo:+.2%}, {hi:+.2%}]"} | {"—" if j == i1 else verdict(lo, hi)} | '
                     f'{S3["A"][:, j].mean():.4f} | {S3["B"][:, j].mean():.4f} | {np.std(single[lam], ddof=1):.4f} |')
        r6, pick = regret(S6, i1, blocks, rng)
        q95 = float(np.percentile(r6, 95))
        ra, _ = regret(S3['A'], i1, blocks, rng)
        rb, _ = regret(S3['B'], i1, blocks, rng)
        L += ['', '### 遗憾（事先定的标准：95 分位 ≤ 0.5%）', '',
              f'- 6 个种子：中位数 {np.median(r6):.3%}，95 分位 {q95:.3%}，最大 {r6.max():.3%}；重抽样里 λ = 1 就是最优的比例 {np.mean(pick == i1):.1%}。',
              f'- 同样 5 个 λ、只用 3 个种子：种子 A 95 分位 {np.percentile(ra, 95):.3%}，种子 B 95 分位 {np.percentile(rb, 95):.3%}。',
              f'- **判定：{"等效" if q95 <= MARGIN else "未达到等效标准"}**（{q95:.3%} {"≤" if q95 <= MARGIN else ">"} {MARGIN:.1%}）。', '',
              '### 交叉验证（一组种子选 λ，另一组评分）', '']
        for fit, ev in (('A', 'B'), ('B', 'A')):
            j = int(S3[fit].mean(0).argmin())
            d, lo, hi = boot(S3[ev][:, i1], S3[ev][:, j], blocks, rng)
            L.append(f'- 种子 {fit} 选出 λ = {lams[j]:g}，在种子 {ev} 上相对 λ = 1：{d:+.2%} [{lo:+.2%}, {hi:+.2%}]，{verdict(lo, hi)}。')
        f, p = stats.f_oneway(*[single[lam] for lam in lams])
        x = np.concatenate([[lam] * 6 for lam in lams])
        lr = stats.linregress(x, np.concatenate([single[lam] for lam in lams]))
        L += ['', '### 方差分析（6 个单种子总分）', '',
              f'- λ 的影响：F = {f:.2f}，p = {p:.3f}（{"超过" if p < .05 else "没有超过"}种子噪声）。',
              f'- 线性趋势：每增加 1，S 变化 {lr.slope:+.5f}，p = {lr.pvalue:.3f}。', '']
    # ensemble level
    def ens(qt, qg):
        return np.stack([total((qt[a] + qg[b]) / 2, y) for a in lams for b in lams], axis=1)
    T, G = Q['transformer'], Q['gru']
    E6 = ens(T['q6'], G['q6'])
    E3 = {g: ens(T['q3'][g], G['q3'][g]) for g in SEEDS}
    base = i1 * len(lams) + i1
    pair = lambda j: (lams[j // len(lams)], lams[j % len(lams)])
    r6, pick = regret(E6, base, blocks, rng)
    q95 = float(np.percentile(r6, 95))
    best = int(E6.mean(0).argmin())
    d, lo, hi = boot(E6[:, base], E6[:, best], blocks, rng)
    ra, _ = regret(E3['A'], base, blocks, rng)
    rb, _ = regret(E3['B'], base, blocks, rng)
    L += ['## 集成（5 × 5 组 (λ_T, λ_G)，w = 0.5）', '',
          f'- (1, 1) 的 S = {E6[:, base].mean():.4f}；最优 {pair(best)}，S = {E6[:, best].mean():.4f}，相对 (1, 1)：{d:+.2%} [{lo:+.2%}, {hi:+.2%}]，{verdict(lo, hi)}。',
          f'- 遗憾：6 个种子中位数 {np.median(r6):.3%}，95 分位 {q95:.3%}，最大 {r6.max():.3%}；只用 3 个种子时 95 分位：A {np.percentile(ra, 95):.3%}，B {np.percentile(rb, 95):.3%}。',
          f'- **判定：{"等效" if q95 <= MARGIN else "未达到等效标准"}**（{q95:.3%} {"≤" if q95 <= MARGIN else ">"} {MARGIN:.1%}）。']
    for fit, ev in (('A', 'B'), ('B', 'A')):
        j = int(E3[fit].mean(0).argmin())
        d, lo, hi = boot(E3[ev][:, base], E3[ev][:, j], blocks, rng)
        L.append(f'- 种子 {fit} 选出 {pair(j)}，在种子 {ev} 上相对 (1, 1)：{d:+.2%} [{lo:+.2%}, {hi:+.2%}]，{verdict(lo, hi)}。')
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'lambda_equivalence_6seeds.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L)); print(f'\n→ {out}')


if __name__ == '__main__':
    main()
