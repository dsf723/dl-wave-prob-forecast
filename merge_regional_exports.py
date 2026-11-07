"""把两个模型导出的 30 天预报表合并成一张区域预报表，并逐站给出检查结果。

用于韩国站：有训练数据的站用"马罗岛 + 13 个韩国站"模型（KKR13，带站点独热）；没有训练数据的站（22188、22189）
用海峡海区模型（STRAIT，无站点独热，能迁移到同海区新站）。--from-second 指定哪些站取第二个表。
逐站检查（有实测的时刻）：P10–P90 覆盖率、低于 P10 / 高于 P90 的比例、P50 平均绝对误差与 GEFS 原始 P50 的误差、
未来 30 天平均波浪能相对气候态的技巧。
用法：python merge_regional_exports.py <第一个 _30d.csv> <第二个 _30d.csv> --from-second KR22188 KR22189 --out <合并后的 csv>
"""
import argparse
import sys

import numpy as np
import pandas as pd


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('first')
    p.add_argument('second')
    p.add_argument('--from-second', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    f1, f2 = pd.read_csv(a.first), pd.read_csv(a.second)
    for f, name in ((f1, a.first), (f2, a.second)):
        if 'model_source' not in f:                         # 已合并过的表保留原来的来源（可多次合并）
            f['model_source'] = name.split('/')[-1].split('\\')[-1]
    keep = pd.concat([f1[~f1.site_id.astype(str).isin(a.from_second)], f2[f2.site_id.astype(str).isin(a.from_second)]], ignore_index=True)
    keep = keep.sort_values(['site_id', 'issue_time', 'lead_hours'], kind='stable')
    keep.to_csv(a.out, index=False, encoding='utf-8-sig')
    e1 = a.first.replace('.csv', '_period_energy.csv'); e2 = a.second.replace('.csv', '_period_energy.csv')
    try:
        p1, p2 = pd.read_csv(e1), pd.read_csv(e2)
        pe = pd.concat([p1[~p1.site_id.astype(str).isin(a.from_second)], p2[p2.site_id.astype(str).isin(a.from_second)]])
        pe.to_csv(a.out.replace('.csv', '_period_energy.csv'), index=False, encoding='utf-8-sig')
    except FileNotFoundError:
        pe = None
    print(f'合并 {len(keep)} 行，{keep.site_id.nunique()} 个站 → {a.out}\n')
    print('| 站点 | 来源模型 | 有实测时刻 | P10–P90 覆盖 | 低于 P10 | 高于 P90 | P50 误差 m | GEFS 原始 P50 误差 m | 30 天平均波浪能技巧 |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    m = keep[keep.source != 'climatology'].dropna(subset=['Hs_observed'])
    for site, g in m.groupby('site_id'):
        y = g.Hs_observed
        skill = ''
        if pe is not None:
            q = pe[(pe.site_id == site) & (pe.observed_fraction >= .8)]
            if len(q):
                ef = (q.WEF_next30d_mean_kWm - q.WEF_observed_next30d_kWm).abs().mean()
                ec = (q.WEF_next30d_clim_kWm - q.WEF_observed_next30d_kWm).abs().mean()
                skill = f'{1 - ef / ec:+.2f}（{len(q)} 次）'
        print(f'| {site} | {g.model_source.iloc[0][:28]} | {len(g)} | {((y >= g.Hs_P10) & (y <= g.Hs_P90)).mean():.0%} | '
              f'{(y < g.Hs_P10).mean():.0%} | {(y > g.Hs_P90).mean():.0%} | {(g.Hs_P50 - y).abs().mean():.3f} | '
              f'{(g.GEFS_raw_P50 - y).abs().mean():.3f} | {skill} |')


if __name__ == '__main__':
    main()
