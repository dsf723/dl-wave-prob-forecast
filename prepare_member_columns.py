"""11 个集合成员各自的波高（10-31，对成员做 PCA 用）：不改原有 forecasts.csv，另存一份 members.csv.gz 放在同一目录，
配置里写 "member_features": true 时按（起报, 时刻）并进预报表。列名 m00（c00）、m01–m10（p01–p10）。
目录：processed/site_<ID>_reforecast/ 与 processed/site_<ID>_era5pre/。
用法：python prepare_member_columns.py [G K H]
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import prepare_reforecast_dataset as prep        # noqa: E402

MEMBERS = ['c00'] + [f'p{i:02d}' for i in range(1, 11)]
COLS = [f'm{i:02d}' for i in range(11)]
FMT = '%Y-%m-%dT%H:%M:%SZ'


def run(sid):
    rf = prep.load_reforecast(prep.SITES[sid]['buoy']['station'])
    rf = rf[rf['member'].isin(MEMBERS)]
    wide = rf.pivot(index=['issue_time', 'valid_time'], columns='member', values='hs').reindex(columns=MEMBERS)
    wide.columns = COLS
    wide = wide.reset_index()
    wide['issue_time'], wide['valid_time'] = wide['issue_time'].dt.strftime(FMT), wide['valid_time'].dt.strftime(FMT)
    for sub in ('reforecast', 'era5pre'):
        d = os.path.join(ROOT, 'processed', f'site_{sid}_{sub}')
        keys = pd.read_csv(os.path.join(d, 'forecasts.csv'), usecols=['issue_time', 'valid_time'])
        out = keys.merge(wide, on=['issue_time', 'valid_time'], how='left')
        miss = out[COLS].isna().all(axis=1).mean()
        out.round(4).to_csv(os.path.join(d, 'members.csv.gz'), index=False)
        print(f'{sid} {sub}：{len(out)} 行，全缺成员的行 {miss:.2%}，某个成员缺的比例 {out[COLS].isna().any(axis=1).mean():.2%}', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    for s in (sys.argv[1:] or ['G', 'K', 'H']):
        run(s)
