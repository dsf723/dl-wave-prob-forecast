"""大浪上尾重训方案对比（验证集）：各方案相对本机基准的 CRPS、阈值加权 CRPS（2.5 m、3.5 m）、
大浪条件 PIT（Allen 2023）、报'超 2.5 m 概率 > 0.8'时的实际发生率、P10–P90 覆盖率；按 block_id 配对自助 2000 次。
事先定的采纳标准：twCRPS 2.5 m 显著变好，且全部 CRPS 不显著变差。
用法：python compare_tail_variants.py [--base GKH_h15_local_transformer_pretrain] [--variants ...]
"""
import argparse
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
from wavecast.postproc import conditional_pit, cdf_at                        # noqa: E402

VARIANTS = ['GKH_h15_tw3_transformer_pretrain', 'GKH_h15_tw2thr_transformer_pretrain', 'GKH_h15_deg24_transformer_pretrain']


def tw(q, y, t):
    return score(np.maximum(q, t), np.maximum(y, t))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='GKH_h15_local_transformer_pretrain')
    p.add_argument('--variants', nargs='+', default=VARIANTS)
    a = p.parse_args()
    run = os.path.join(ROOT, 'model_runs', a.base)
    best = json.load(open(os.path.join(run, 'selection.json'), encoding='utf-8'))['best']
    model, info = next(iter(best.items()))
    cs = pd.read_csv(os.path.join(run, 'validation', 'case_scores.csv'), usecols=['model', 'truth', 'block_id', 'site_id'])
    cs = cs[cs.model == f"{model}_{info['ablation']}"].reset_index(drop=True)
    y, blocks = cs.truth.to_numpy(float), cs.block_id.to_numpy()
    runs = [a.base] + [v for v in a.variants if os.path.exists(os.path.join(ROOT, 'model_runs', v, 'selection.json'))]
    qs = {r: seed_mean(r) for r in runs}
    rng = np.random.default_rng(0)
    big = y > 2.5
    rows = []
    for r, q in qs.items():
        c, _ = conditional_pit(q, y, 2.5, rng)
        pf = 1 - cdf_at(q, 2.5)
        hi = pf > .8
        rows.append(dict(运行=r.replace('_transformer_pretrain', ''), CRPS=score(q, y).mean(), twCRPS_25=tw(q, y, 2.5).mean(),
                         twCRPS_35=tw(q, y, 3.5).mean(), 覆盖80=((y >= q[:, 9]) & (y <= q[:, 89])).mean(),
                         条件PIT最高一成=(c > .9).mean(), 报gt08_预报=pf[hi].mean(), 报gt08_实际=big[hi].mean(), 报gt08_个数=int(hi.sum())))
    res = pd.DataFrame(rows)
    lines = ['| ' + ' | '.join(res.columns) + ' |', '|' + '---|' * len(res.columns)]
    lines += ['| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r) + ' |' for _, r in res.iterrows()]
    lines += ['', '| 方案（相对本机基准） | 指标 | 变化 | 95% 区间 | 判断 |', '|---|---|---:|---|---|']
    q0 = qs[a.base]
    for r in runs[1:]:
        q = qs[r]; v = {}
        for name, f in (('CRPS', score), ('twCRPS 2.5 m', lambda q, y: tw(q, y, 2.5)), ('twCRPS 3.5 m', lambda q, y: tw(q, y, 3.5))):
            d, lo, hi = boot(f(q0, y), f(q, y), blocks, rng)
            v[name] = '显著变好' if hi < 0 else ('显著变差' if lo > 0 else '不显著')
            lines.append(f'| {r.replace("_transformer_pretrain", "")} | {name} | {d:+.2%} | [{lo:+.2%}, {hi:+.2%}] | {v[name]} |')
        ok = v['twCRPS 2.5 m'] == '显著变好' and v['CRPS'] != '显著变差'
        lines.append(f'| {r.replace("_transformer_pretrain", "")} | **按事先标准** | | | {"采纳" if ok else "不采纳"} |')
    out = os.path.join(ROOT, 'results', 'postproc', 'tail_variants.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
