"""气象署海象环境资讯平台“资料浮标观测历史资料”（波浪）的完整度检查（5 站 74 个年文件）。

输入：raw/cwa_buoy_history/<站>_<年>_wave.zip（每个 zip 一个 CSV：st,yyyymmddhhmi,H(cm),T=尖峰周期(0.1 s),Tmean=平均周期(0.1 s),D(度)；时间为 UTC，11-04 与 Copernicus 同站对过）。
内容：每站每年的逐时完整度（按应有小时数）、Hs / Tp / Tmean 有效比例与均值、最大 Hs；不改任何数据，只描述。
输出：results/cwa_buoys/coverage.md
用法：python check_cwa_buoys.py
"""
import glob
import io
import os
import sys
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'raw', 'cwa_buoy_history')
OUT = os.path.join(ROOT, 'results', 'cwa_buoys', 'coverage.md')
NAMES = {'46757B': '新竹', 'C6W08': '马祖', 'C6W10': '七美', 'C5W09': '东吉岛', 'C6F01': '台中'}


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode('utf-8-sig')
    lines = text.splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith('st,'))
    d = pd.read_csv(io.StringIO('\n'.join(lines[k:])), dtype={'yyyymmddhhmi': str})
    d['t'] = pd.to_datetime(d['yyyymmddhhmi'], format='%Y%m%d%H%M')
    for c in ('H', 'T', 'Tmean', 'D'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.drop_duplicates('t')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, '*_wave.zip'))):
        st, year = os.path.basename(path).split('_')[:2]
        d = read_zip(path)
        year = int(year)
        start, end = pd.Timestamp(f'{year}-01-01'), min(pd.Timestamp(f'{year + 1}-01-01'), d.t.max().ceil('D'))
        hours = (end - start) / pd.Timedelta('1h')
        h = d.H.where(d.H > 0) / 100
        rows.append(dict(st=st, year=year, n=len(d), cover=h.notna().sum() / hours, tp=d['T'].notna().mean(), tm=d.Tmean.notna().mean(),
                         hs=h.mean(), hmax=h.max(), tpm=d['T'].mean() / 10, tmm=d.Tmean.mean() / 10, first=d.t.min(), last=d.t.max()))
    r = pd.DataFrame(rows)
    L = ['# 气象署浮标历史资料（波浪）完整度', '', '完整度 = 有效 Hs 的小时数 ÷ 当年应有小时数（当年最后一天按资料最后时刻截止）。Tp、Tmean 列为有 Hs 的行里周期有值的比例。', '']
    for st, g in r.groupby('st', sort=False):
        L += [f'## {NAMES.get(st, st)} {st}', '', f'资料 {g["first"].min():%Y-%m-%d} 至 {g["last"].max():%Y-%m-%d}；回算重叠年（2000–2019）平均完整度 {g[g.year.between(2000, 2019)].cover.mean():.0%}，2020 年后 {g[g.year >= 2020].cover.mean():.0%}。', '',
              '| 年 | 行数 | 完整度 | Tp 有值 | Tmean 有值 | 平均 Hs m | 最大 Hs m | 平均 Tp s | 平均 Tmean s |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        L += [f'| {x.year} | {x.n:,} | {x.cover:.0%} | {x.tp:.0%} | {x.tm:.0%} | {x.hs:.2f} | {x.hmax:.2f} | {x.tpm:.1f} | {x.tmm:.1f} |' for x in g.itertuples()]
        L.append('')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
