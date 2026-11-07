"""比较两次训练在验证集上的成绩（只比两次都有的站点、同一批起报时刻），看新改动有没有让预报更准。

读取 <run>/validation/case_scores.csv 里各自"冻结的候选模型"（model 列 = selection.json 选出的模型名，
即各种子平均后的预报），按 站点 × 时刻 对齐后比较 CRPS：全部、各站点、台风期、预见期段，
并用按 block_id（事件 / 4 周背景块）的配对块自助法（2000 次）给出变化的 95% 区间，区间不含 0 才算显著。
只看验证集，测试集不碰。
用法：python compare_runs.py model_runs/GKH_h15_transformer_pretrain model_runs/GKHH2_h15_transformer_pretrain
"""
import json
import os
import sys

import numpy as np
import pandas as pd

KEY = ['site_id', 'issue_time', 'valid_time']


def load(run):
    best = json.load(open(os.path.join(run, 'selection.json'), encoding='utf-8'))['best']
    model = next(iter(best))
    d = pd.read_csv(os.path.join(run, 'validation', 'case_scores.csv'),
                    usecols=KEY + ['model', 'crps', 'lead_band', 'typhoon', 'block_id', 'covered80'])
    d = d[d.model == model].drop(columns='model')
    # 不同机器 / 版本写出的时间字符串可能不同（如带小数秒），按时间值而不是字符串对齐
    for c in ('issue_time', 'valid_time'):
        d[c] = pd.to_datetime(d[c], utc=True, format='mixed')
    d['site_id'] = d['site_id'].astype(str)
    return d, model


def change(a, b, blocks, rng, n=2000):
    """Relative CRPS change of b vs a and a paired block-bootstrap 95% interval."""
    df = pd.DataFrame({'a': a, 'b': b, 'blk': blocks}).groupby('blk').agg(['sum', 'count'])
    sa, sb, cnt = df[('a', 'sum')].to_numpy(), df[('b', 'sum')].to_numpy(), df[('a', 'count')].to_numpy()
    rel = sb.sum() / sa.sum() - 1
    idx = rng.integers(0, len(sa), (n, len(sa)))
    boot = sb[idx].sum(1) / sa[idx].sum(1) - 1
    return rel, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    (a, ma), (b, mb) = load(sys.argv[1]), load(sys.argv[2])
    m = a.merge(b[KEY + ['crps', 'covered80']], on=KEY, suffixes=('_a', '_b'))
    sites = sorted(m.site_id.unique())
    if m.empty:
        raise SystemExit(f'两次运行没有对齐的时刻。示例键：\nA {a[KEY].head(2).to_string()}\nB {b[KEY].head(2).to_string()}')
    print(f'A = {sys.argv[1]}（{ma}）\nB = {sys.argv[2]}（{mb}）\n共同站点 {sites}，对齐 {len(m)} 个时刻'
          f'（A 原有 {len(a)}，B 原有 {len(b)}；B 多出的站点不参与比较）\n')
    rng = np.random.default_rng(0)
    groups = [('全部', np.ones(len(m), bool))] + [(f'站点 {s}', (m.site_id == s).to_numpy()) for s in sites]
    groups += [('台风期', m.typhoon.astype(bool).to_numpy())]
    groups += [(f'预见期 {x}', (m.lead_band == x).to_numpy()) for x in sorted(m.lead_band.unique())]
    print('| 分组 | 时刻数 | A 的 CRPS | B 的 CRPS | 变化 | 95% 区间 | 显著 | A 覆盖率 | B 覆盖率 |')
    print('|---|---:|---:|---:|---:|---|---|---:|---:|')
    for name, k in groups:
        if k.sum() == 0:
            continue
        g = m[k]
        rel, lo, hi = change(g.crps_a.to_numpy(), g.crps_b.to_numpy(), g.block_id.to_numpy(), rng)
        sig = '是' if hi < 0 or lo > 0 else '否'
        print(f'| {name} | {k.sum()} | {g.crps_a.mean():.4f} | {g.crps_b.mean():.4f} | {rel:+.1%} | '
              f'[{lo:+.1%}, {hi:+.1%}] | {sig} | {g.covered80_a.mean():.0%} | {g.covered80_b.mean():.0%} |')
    print('\n变化为负 = B 更好（CRPS 越小越好）。覆盖率 = 实测落在 P10–P90 的比例，理想值 80%。')


if __name__ == '__main__':
    main()
