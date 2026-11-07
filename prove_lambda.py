"""用多种方法检验损失权重 λ = 1（只用验证集，不碰测试集）。损失 = CRPS + λ·twCRPS(2.5 m)。
点集：网格 0、0.5、1、2、4 与贝叶斯优化 12 次试验，共 17 个 λ，每个都是 3 个种子分位数平均（种子 11、23、37）。
  1. 等效性检验：块自助 2000 次，每次重选最优 λ，遗憾 = S(1) ÷ min S(λ) − 1；95 分位 ≤ 0.5% 判等效（标准事先定）。
  2. 样本外检验：留一站、留一年、5 折分块，在其余数据上选 λ，在留出数据上与 λ = 1 比较。
  3. 独立训练检验：选出的最优 λ 用新种子 41、53、67 重训，与同种子的 λ = 1 配对比较；并看只换种子时 λ = 1 自身变化多少。
  4. 集成层面：17 × 17 组 (λ_T, λ_G)、w = 0.5，看 (1, 1) 的遗憾（补充）。
用法：python prove_lambda.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from route_models import seed_mean, boot                                     # noqa: E402
from wavecast.metrics import score                                           # noqa: E402

MARGIN = .005
GRID = [(0., '0'), (.5, '05'), (1., '1'), (2., '2'), (4., '4')]
OUT = os.path.join(ROOT, 'results', 'weights')


def total(q, y):
    return score(q, y) + score(np.maximum(q, 2.5), np.maximum(y, 2.5))


def verdict(lo, hi):
    return '显著变好' if hi < 0 else ('显著变差' if lo > 0 else '不显著')


def quantiles_by_lambda(m):
    """{λ: 3-seed quantiles} for one model: grid runs and Bayesian-optimisation trials."""
    out = {lam: seed_mean(f'GKH_h15_lam{tag}_{m}_pretrain') for lam, tag in GRID}
    d = os.path.join(ROOT, 'model_runs', f'GKH_h15_bo3_lambda_{m}')
    trials = pd.read_csv(os.path.join(d, 'trials.csv'))
    for _, t in trials.iterrows():
        path = os.path.join(d, f'trial{int(t.number)}_val.npz')
        if os.path.exists(path):
            out[float(t.params_lam)] = np.load(path)['quantiles']
    return dict(sorted(out.items()))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cs = pd.read_csv(os.path.join(ROOT, 'model_runs', 'GKH_h15_solo_base_transformer_pretrain', 'validation', 'case_scores.csv'),
                     usecols=['model', 'truth', 'block_id', 'site_id', 'issue_time'])
    cs = cs[cs.model == 'transformer_season'].reset_index(drop=True)
    y, blocks = cs.truth.to_numpy(float), cs.block_id.to_numpy()
    site, years = cs.site_id.to_numpy(), cs.issue_time.str[:4].to_numpy()
    key = pd.Series(site + '_' + cs.issue_time.str[:7].to_numpy())
    rng = np.random.default_rng(0)
    uk = np.array(sorted(key.unique()))
    fold5 = key.map(dict(zip(uk, rng.permutation(len(uk)) % 5))).to_numpy()
    L = ['# 损失权重 λ 的检验（验证集）', '',
         f'损失 = CRPS + λ·twCRPS(2.5 m)，总分 S = CRPS + twCRPS(2.5 m)；{len(y)} 个时刻；全部是本机模型，3 个种子分位数平均。', '']
    Q = {}
    for m in ['transformer', 'gru']:
        qs = Q[m] = quantiles_by_lambda(m)
        lams = np.array(list(qs))
        S = np.stack([total(q, y) for q in qs.values()], axis=1)
        i1 = int(np.where(lams == 1)[0][0])
        mean = S.mean(0)
        ib = int(mean.argmin())
        d, lo, hi = boot(S[:, i1], S[:, ib], blocks, rng)
        bs = pd.DataFrame(S).groupby(blocks).sum().to_numpy()
        idx = rng.integers(0, len(bs), (2000, len(bs)))
        res = bs[idx].sum(1)
        regret = res[:, i1] / res.min(1) - 1
        l_star = lams[res.argmin(1)]
        q95 = float(np.percentile(regret, 95))
        L += [f'## {m}', '', f'共 {len(lams)} 个 λ（{lams.min():g}～{lams.max():g}）；S(λ = 1) = {mean[i1]:.4f}；全部 λ 的 S 从 {mean.min():.4f} 到 {mean.max():.4f}。', '',
              '### 1. 等效性检验（事先定的标准：遗憾 95 分位 ≤ 0.5%）', '',
              f'- 验证集最优 λ = {lams[ib]:.3f}，S = {mean[ib]:.4f}，相对 λ = 1：{d:+.2%} [{lo:+.2%}, {hi:+.2%}]，{verdict(lo, hi)}。',
              f'- 每次重抽样重选最优：最优 λ 的 2.5–97.5 分位 [{np.percentile(l_star, 2.5):.3f}, {np.percentile(l_star, 97.5):.3f}]。',
              f'- λ = 1 比当次最优差：中位数 {np.median(regret):.3%}，95 分位 {q95:.3%}，最大 {regret.max():.3%}。',
              f'- **判定：{"等效" if q95 <= MARGIN else "未达到等效标准"}**（{q95:.3%} {"≤" if q95 <= MARGIN else ">"} {MARGIN:.1%}）。', '',
              '### 2. 样本外检验（在留出数据上相对 λ = 1）', '', '| 留出方式 | 相对 λ = 1 | 判断 | 各折选出的 λ |', '|---|---|---|---|']
        for name, folds in (('留一站', site), ('留一年', years), ('5 折分块', fold5)):
            out, chosen = np.zeros(len(y)), []
            for k in np.unique(folds):
                j = int(S[folds != k].mean(0).argmin())
                out[folds == k] = S[folds == k, j]
                chosen.append(lams[j])
            d, lo, hi = boot(S[:, i1], out, blocks, rng)
            L.append(f'| {name} | {d:+.3%} [{lo:+.3%}, {hi:+.3%}] | {verdict(lo, hi)} | {", ".join(f"{x:.3g}" for x in chosen)} |')
        L += ['', '### 3. 独立训练检验（新种子 41、53、67）', '']
        s2, star = f'GKH_h15_why6_base_s2_{m}_pretrain', f'GKH_h15_lamstar_s2_{m}_pretrain'
        if os.path.exists(os.path.join(ROOT, 'model_runs', star, 'selection.json')):
            a, b = total(seed_mean(s2), y), total(seed_mean(star), y)
            d, lo, hi = boot(a, b, blocks, rng)
            ds, los, his = boot(S[:, i1], a, blocks, rng)
            L += [f'- 最优 λ = {json.load(open(os.path.join(ROOT, "model_runs", f"GKH_h15_bo3_lambda_{m}", "best_lambda.json"), encoding="utf-8"))["best_trial_lambda"]:.3f}'
                  f' 换新种子：S = {b.mean():.4f}；同种子 λ = 1：S = {a.mean():.4f}；相对 λ = 1：{d:+.2%} [{lo:+.2%}, {hi:+.2%}]，{verdict(lo, hi)}。',
                  f'- 只换种子、λ = 1 不变：S 从 {S[:, i1].mean():.4f} 变为 {a.mean():.4f}，{ds:+.2%} [{los:+.2%}, {his:+.2%}]。这是训练随机性的大小。']
        else:
            L.append('- 还没有训练完。')
        L.append('')
    # ensemble level
    lt, lg = list(Q['transformer']), list(Q['gru'])
    E = np.empty((len(y), len(lt) * len(lg)), dtype=np.float32)
    for i, a in enumerate(lt):
        for j, b in enumerate(lg):
            E[:, i * len(lg) + j] = total((Q['transformer'][a] + Q['gru'][b]) / 2, y)
    base = lt.index(1.) * len(lg) + lg.index(1.)
    best = int(E.mean(0).argmin())
    d, lo, hi = boot(E[:, base], E[:, best], blocks, rng)
    bs = pd.DataFrame(E).groupby(blocks).sum().to_numpy()
    res = bs[rng.integers(0, len(bs), (2000, len(bs)))].sum(1)
    regret = res[:, base] / res.min(1) - 1
    L += ['## 4. 集成层面（补充）', '',
          f'- {len(lt)} × {len(lg)} 组 (λ_T, λ_G)、w = 0.5：(1, 1) 的 S = {E[:, base].mean():.4f}；最优 (λ_T, λ_G) = ({lt[best // len(lg)]:.3f}, {lg[best % len(lg)]:.3f})，S = {E[:, best].mean():.4f}，相对 (1, 1)：{d:+.2%} [{lo:+.2%}, {hi:+.2%}]，{verdict(lo, hi)}。',
          f'- 每次重抽样重选最优：(1, 1) 的遗憾中位数 {np.median(regret):.3%}，95 分位 {np.percentile(regret, 95):.3%}，最大 {regret.max():.3%}。', '',
          '说明：点集越大、噪声越大，"当次最优"越容易碰巧偏好，所以遗憾偏保守（对 λ = 1 不利）。各 λ 的点估计在定标准之前已看过，所以不是完全盲；0.5% 的界限沿用集成权重的检验，不是看结果后定的。']
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'lambda_equivalence.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L)); print(f'\n→ {out}')


if __name__ == '__main__':
    main()
