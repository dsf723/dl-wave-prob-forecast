"""逐站看订正效果：一次训练运行的验证集上，模型与 GEFS 原始集合、气候态的 CRPS 和覆盖率（每站一行）。

读取 <run>/validation/case_scores.csv：冻结的候选模型（selection.json 选出的模型名）、raw_ensemble（GEFS 原始集合）、
climatology（训练年逐月气候态），同一批时刻。--sites 只看指定站点（如韩国海峡站）。
用法：python site_skill.py model_runs/KKR13_h15_transformer_pretrain [--sites KR22103 KR22104 KR22188 KR22189]
"""
import argparse
import json
import os
import sys

import pandas as pd


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--sites', nargs='*')
    a = p.parse_args()
    model = next(iter(json.load(open(os.path.join(a.run, 'selection.json'), encoding='utf-8'))['best']))
    d = pd.read_csv(os.path.join(a.run, 'validation', 'case_scores.csv'), usecols=['site_id', 'model', 'crps', 'covered80', 'typhoon'])
    d['site_id'] = d.site_id.astype(str)
    if a.sites:
        d = d[d.site_id.isin(a.sites)]
    g = d[d.model.isin([model, 'raw_ensemble', 'climatology'])].groupby(['site_id', 'model']).agg(crps=('crps', 'mean'), cov=('covered80', 'mean'), n=('crps', 'size')).unstack('model')
    print(f'运行 {a.run}（模型 {model}，验证集）')
    print('| 站点 | 时刻数 | 模型 CRPS | GEFS 原始 CRPS | 气候态 CRPS | 相对 GEFS | 相对气候态 | 模型覆盖率 | GEFS 覆盖率 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for site, r in g.iterrows():
        m, raw, cl = r[('crps', model)], r[('crps', 'raw_ensemble')], r[('crps', 'climatology')]
        print(f"| {site} | {int(r[('n', model)])} | {m:.4f} | {raw:.4f} | {cl:.4f} | {m / raw - 1:+.1%} | {m / cl - 1:+.1%} | "
              f"{r[('cov', model)]:.0%} | {r[('cov', 'raw_ensemble')]:.0%} |")


if __name__ == '__main__':
    main()
