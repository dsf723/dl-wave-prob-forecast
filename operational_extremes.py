"""独立业务检验的补充诊断（10-28）：大浪条件 PIT、事后实况台风 / 寒潮子集、逐预见日 CRPSS、Hersbach 分解。

只读 evaluate_operational.py 已写出的结果，不重新预测：
- quantiles.npz：各系统的 99 个分位数；
- cases.csv：实测；
- case_scores.csv：取其中的事件标记。

方法与 postproc_diagnostics.py --split test 相同，便于和 2017–2019 测试集并列：
- 条件 PIT（Allen 2023），阈值为 2.5 m 与本站训练年 90% 分位；
- 事件子集按 block_id 配对自助 2000 次。
"按实测挑子集"的比较受预报者困境影响（Lerch 2017），大浪的恰当比较以全部样本上的 twCRPS 为准。

用法：python operational_extremes.py <结果目录> [<结果目录> ...]
      结果写到 results/operational/<目录名>_extremes.md
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
from project_paths import resolve_config                                     # noqa: E402
from route_models import boot                                                # noqa: E402
from wavecast.experiment import load_sites                                   # noqa: E402
from wavecast.metrics import score                                           # noqa: E402
from wavecast.postproc import hersbach, conditional_pit                      # noqa: E402

OUT = os.path.join(ROOT, 'results', 'operational')
LABEL = {'ensemble': '集成（最终方案）', 'transformer': 'Transformer', 'gru': 'GRU', 'ensemble_cqr': '集成 + 共形校正',
         'emos_ln': 'EMOS 对数正态', 'emos_tn': 'EMOS 截断正态', 'raw_ensemble': 'GEFS 业务原始集合', 'climatology': '气候态'}
KEY = ['site_id', 'issue_time', 'valid_time']


def tw(q, y, t):
    return score(np.maximum(q, t), np.maximum(y, t))


def md(df, fmt='{:.3f}'):
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + ' |')
    return '\n'.join(lines)


def site_q90(run):
    cfg = json.load(open(resolve_config(os.path.join(ROOT, 'model_configs', f'{run}.json')), encoding='utf-8'))
    sites, _ = load_sites(cfg)
    return {str(c['site_id']): float(obs.loc[obs.index.year.isin(c['train_years']), cfg['target']].quantile(.9))
            for c, obs, _ in sites}


def diagnose(d, q90, t):
    q = dict(np.load(os.path.join(d, 'quantiles.npz')))
    cases = pd.read_csv(os.path.join(d, 'cases.csv'), dtype={'site_id': str})
    flags = pd.read_csv(os.path.join(d, 'case_scores.csv'), usecols=KEY + ['model', 'typhoon', 'cold_surge'], dtype={'site_id': str})
    flags = flags[flags.model == 'ensemble'].drop(columns='model').drop_duplicates(KEY)
    cases = cases.merge(flags, on=KEY, how='left', validate='one_to_one')
    if cases.typhoon.isna().any():
        raise SystemExit(f'{d}：有 {cases.typhoon.isna().sum()} 个时刻在 case_scores.csv 里找不到事件标记')
    y = cases.truth.to_numpy(float)
    systems = {k: q[k] for k in LABEL if k in q}
    if any(len(v) != len(y) for v in systems.values()):
        raise SystemExit(f'{d}：分位数行数与 cases.csv 不一致')
    event = (cases.typhoon.astype(bool) | cases.cold_surge.astype(bool)).to_numpy()
    blocks = cases.block_id.astype(str).to_numpy()
    site_t = cases.site_id.map(q90).to_numpy(float)
    rng = np.random.default_rng(0)
    rep = [f'# 业务独立检验补充诊断：{os.path.basename(os.path.normpath(d))}', '',
           f'{len(y)} 个时刻；实测 > {t} m 的 {int((y > t).sum())} 个，实测 > 本站训练年 90% 分位的 {int((y > site_t).sum())} 个'
           f'（阈值 {", ".join(f"{s} {v:.2f} m" for s, v in q90.items())}）；'
           f'事后实况台风 / 寒潮 {int(event.sum())} 个（台风 {int(cases.typhoon.astype(bool).sum())}、寒潮 {int(cases.cold_surge.astype(bool).sum())}）。', '']

    # 1. conditional PIT
    rows = []
    for name, thr in [(f'实测 > {t} m', np.full(len(y), t)), ('实测 > 本站训练年 90% 分位', site_t)]:
        for k, v in systems.items():
            if k == 'climatology':
                continue
            vs = []
            for s in np.unique(thr):
                m = thr == s
                c, _ = conditional_pit(v[m], y[m], s, rng)
                vs.append(c)
            c = np.concatenate(vs)
            h = np.histogram(c, bins=10, range=(0, 1))[0] / max(len(c), 1)
            above = (y > thr)
            rows.append({'样本': name, '系统': LABEL[k], 'n': len(c), '最低一成': h[0], '最高一成': h[-1],
                         '平均条件PIT': float(c.mean()), '实测高于P90': float((y[above] > v[above, 89]).mean())})
    rep += ['## 1. 大浪条件 PIT（Allen 2023）', '',
            '理想值：每格 0.1，平均 0.5；"实测高于 P90"理想值约 0.1 以下（条件样本里会偏大，只作系统间比较）。'
            '"最高一成"偏大 = 大浪时上尾偏轻。', '', md(pd.DataFrame(rows)), '']

    # 2. twCRPS on all cases (proper) and the observed-event subset
    ens = systems['ensemble']
    rows = []
    for gname, k in [('全部（twCRPS，恰当比较）', np.ones(len(y), bool))]:
        for ref in ('emos_ln', 'raw_ensemble'):
            dd, lo, hi = boot(tw(systems[ref][k], y[k], t), tw(ens[k], y[k], t), blocks[k], rng)
            rows.append(dict(分组=gname, 指标=f'twCRPS {t} m', 对照=LABEL[ref], 变化=f'{dd:+.1%}', 区间=f'[{lo:+.1%}, {hi:+.1%}]',
                             判断='显著变好' if hi < 0 else ('显著变差' if lo > 0 else '不显著')))
    subsets = [('事后实况台风 / 寒潮', event)] + [(f'站点 {s} 台风 / 寒潮', event & (cases.site_id == s).to_numpy())
                                              for s in sorted(cases.site_id.unique())]
    for gname, k in subsets:
        if k.sum() < 50:
            rows.append(dict(分组=gname, 指标='CRPS', 对照='—', 变化=f'n={int(k.sum())}', 区间='样本太少', 判断='—'))
            continue
        for metric, f in (('CRPS', score), (f'twCRPS {t} m', lambda a, b: tw(a, b, t))):
            for ref in ('emos_ln', 'raw_ensemble'):
                dd, lo, hi = boot(f(systems[ref][k], y[k]), f(ens[k], y[k]), blocks[k], rng)
                rows.append(dict(分组=f'{gname}（n={int(k.sum())}）', 指标=metric, 对照=LABEL[ref], 变化=f'{dd:+.1%}',
                                 区间=f'[{lo:+.1%}, {hi:+.1%}]', 判断='显著变好' if hi < 0 else ('显著变差' if lo > 0 else '不显著')))
    rep += ['## 2. 大浪与事件子集：集成相对基准（按 block_id 配对自助 2000 次）', '',
            '按实测挑出的子集受预报者困境影响（Lerch 2017），只作参考；恰当比较是第一行全部样本上的 twCRPS。', '',
            md(pd.DataFrame(rows)), '']
    rows = []
    for k in ('ensemble', 'emos_ln', 'raw_ensemble', 'climatology'):
        rows.append({'系统': LABEL[k], '事件子集 CRPS': score(systems[k][event], y[event]).mean(),
                     '事件子集 twCRPS': tw(systems[k][event], y[event], t).mean(),
                     '事件子集覆盖80': float(((y >= systems[k][:, 9]) & (y <= systems[k][:, 89]))[event].mean())})
    rep += [md(pd.DataFrame(rows), '{:.4f}'), '']

    # 3. CRPSS by lead day
    clim = score(systems['climatology'], y)
    lead = cases.lead_day.to_numpy()
    rows = []
    for k in ('ensemble', 'emos_ln', 'raw_ensemble'):
        s = score(systems[k], y)
        rows.append({'系统': LABEL[k], **{str(int(dday)): 1 - s[lead == dday].mean() / clim[lead == dday].mean()
                                          for dday in sorted(np.unique(lead))}})
    rep += ['## 3. 逐预见日 CRPSS（相对气候态）', '', md(pd.DataFrame(rows), '{:.2f}'), '']

    # 4. Hersbach decomposition
    rows = []
    for k in ('ensemble', 'emos_ln', 'raw_ensemble', 'climatology'):
        h = hersbach(systems[k], y)
        rows.append({'系统': LABEL[k], 'CRPS': h['crps'], '可靠度': h['reliability'], '分辨率': h['resolution'],
                     '不确定度': h['uncertainty']})
    rep += ['## 4. CRPS 分解（Hersbach 2000）', '', '可靠度越小越好，分辨率越大越好。', '', md(pd.DataFrame(rows), '{:.4f}'), '']
    return '\n'.join(rep)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--run', default='GKH_h15_transformer_pretrain', help='取本站训练年 90% 分位所用的配置')
    p.add_argument('--threshold', type=float, default=2.5)
    a = p.parse_args()
    q90 = site_q90(a.run)
    os.makedirs(OUT, exist_ok=True)
    for d in a.dirs:
        rep = diagnose(d, q90, a.threshold)
        path = os.path.join(OUT, os.path.basename(os.path.normpath(d)) + '_extremes.md')
        open(path, 'w', encoding='utf-8').write(rep)
        print(rep, '\n', '→', path, '\n', flush=True)


if __name__ == '__main__':
    main()
