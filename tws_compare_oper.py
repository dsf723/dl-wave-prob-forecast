"""台湾海峡独立业务检验：扩充训练的 TWS2 集成 vs 冻结的 TWS 集成，同一批时刻配对比较（11-04 事先登记）。

读两次 evaluate_operational.py 的输出（cases.csv、quantiles.npz 里的 ensemble），按 站号 × 起报 × 有效时刻 对齐
（冻结模型的站点 ID 是 TW46757B / TWC6W08，新模型是 CW46757B / CWC6W08，同一个浮标，按站号对齐；真值必须一致）。
指标：CRPS 与总分 S = CRPS + twCRPS(2.5 m)；区间：块名去掉站点前缀，同一时段的两站一起抽样，配对块自助 2000 次。
输出：results/operational/tws2_vs_tws_oper.md
用法：python tws_compare_oper.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from route_models import boot                                                # noqa: E402
from wavecast.metrics import score                                           # noqa: E402

OPER = os.path.join(ROOT, 'results', 'operational')
A = 'TWS_h15_transformer_pretrain_ens_oper_TW46757B_TWC6W08'
B = 'TWS2_h15_transformer_pretrain_ens_oper_CW46757B_CWC6W08'


def load(tag):
    c = pd.read_csv(os.path.join(OPER, tag, 'cases.csv'))
    c['station'] = c.site_id.astype(str).str[2:]
    for k in ('issue_time', 'valid_time'):
        c[k] = pd.to_datetime(c[k], utc=True, format='mixed')
    c['block'] = [b[len(s) + 1:] if str(b).startswith(f'{s}_') else str(b) for s, b in zip(c.site_id.astype(str), c.block_id.astype(str))]
    return c, np.load(os.path.join(OPER, tag, 'quantiles.npz'))['ensemble']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (ca, qa), (cb, qb) = load(A), load(B)
    ca['ia'], cb['ib'] = np.arange(len(ca)), np.arange(len(cb))
    m = ca.merge(cb[['station', 'issue_time', 'valid_time', 'ib', 'truth']], on=['station', 'issue_time', 'valid_time'], suffixes=('', '_b'))
    if not np.allclose(m.truth, m.truth_b, atol=1e-3):
        raise SystemExit('两次评分的真值不一致')
    y = m.truth.to_numpy(float)
    xa, xb = qa[m.ia.to_numpy()], qb[m.ib.to_numpy()]
    tw = lambda q: score(np.maximum(q, 2.5), np.maximum(y, 2.5))
    S = {'CRPS': (score(xa, y), score(xb, y)), '总分 S': (score(xa, y) + tw(xa), score(xb, y) + tw(xb))}
    blocks, rng = m.block.to_numpy(), np.random.default_rng(0)
    L = ['# 独立业务检验：扩充训练（TWS2）vs 冻结模型（TWS），同一批时刻', '',
         f'对齐 {len(m):,} 个时刻（冻结模型 {len(ca):,}，新模型 {len(cb):,}），站点 {sorted(m.station.unique())}，块 {len(set(blocks))} 个。负 = 新模型更好。', '',
         '| 分组 | 时刻 | 指标 | 冻结 TWS | 扩充 TWS2 | 变化 | 95% 区间 | 判断 |', '|---|---:|---|---:|---:|---:|---|---|']
    groups = [('全部', np.ones(len(m), bool))] + [(f'站 {s}', (m.station == s).to_numpy()) for s in sorted(m.station.unique())] \
        + [(b, (m.lead_band == b).to_numpy()) for b in sorted(m.lead_band.unique())]
    for g, k in groups:
        for name, (sa, sb) in S.items():
            if g != '全部' and name != 'CRPS':
                continue
            d, lo, hi = boot(sa[k], sb[k], blocks[k], rng)
            L.append(f'| {g} | {k.sum():,} | {name} | {sa[k].mean():.4f} | {sb[k].mean():.4f} | {d:+.1%} | [{lo:+.1%}, {hi:+.1%}] | '
                     f'{"显著变好" if hi < 0 else ("显著变差" if lo > 0 else "不显著")} |')
    out = os.path.join(OPER, 'tws2_vs_tws_oper.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
