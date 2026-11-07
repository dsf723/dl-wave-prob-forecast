"""下载 NDBC 浮标历史逐时资料（标准气象文件 stdmet），用于独立验证。站点见 metadata/sites.json 中 source 为 NDBC 的浮标。

输出：
  raw/observations/ndbc/<站号>h<年>.txt.gz        原始文件（不改动）
  processed/buoy/<站点ID>_<站号>.csv               质控后的逐时序列（UTC）
列：valid_time, hs（WVHT, m）, tp（DPD 谱峰周期, s）, tm02（APD 平均周期, s）, mwd（来向 °）,
    wspd（m/s）, wdir（°）, pres（hPa）, atmp（°C）, wtmp（°C）
NDBC 用 99 / 999 / 9999 表示缺测，这里统一改为空值。周期定义：DPD = 谱峰周期，APD = 平均（零交叉）周期，
都不是 ERA5 的 mwp（能量周期），比较时应分别对应 ERA5 的 pp1d 与 mp2。
用法：python download_buoys.py
"""
import gzip
import io
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
RAW = os.path.join(ROOT, 'raw', 'observations', 'ndbc')
OUT = os.path.join(ROOT, 'processed', 'buoy')
URL = 'https://www.ndbc.noaa.gov/data/historical/stdmet/{st}h{y}.txt.gz'
MISSING = {'WDIR': 999, 'WSPD': 99, 'GST': 99, 'WVHT': 99, 'DPD': 99, 'APD': 99, 'MWD': 999,
           'PRES': 9999, 'ATMP': 999, 'WTMP': 999, 'DEWP': 999, 'VIS': 99, 'TIDE': 99}


def fetch(st, y):
    path = os.path.join(RAW, f'{st}h{y}.txt.gz')
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    req = urllib.request.Request(URL.format(st=st, y=y), headers={'User-Agent': 'research'})
    try:
        b = urllib.request.urlopen(req, timeout=120).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    open(path, 'wb').write(b)
    time.sleep(1)
    return path


def parse(path):
    text = gzip.open(path, 'rt').read().splitlines()
    head = text[0].lstrip('#').split()
    rows = [l.split() for l in text if l and not l.startswith('#')]
    df = pd.DataFrame(rows, columns=head[:len(rows[0])]).apply(pd.to_numeric, errors='coerce')
    yy = 'YY' if 'YY' in df else 'YYYY'
    t = pd.to_datetime(dict(year=df[yy], month=df['MM'], day=df['DD'], hour=df['hh'],
                            minute=df['mm'] if 'mm' in df else 0), errors='coerce')
    df.index = t
    for k, v in MISSING.items():
        if k in df:
            df.loc[df[k] >= v, k] = np.nan
    return df


def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    for s in SITES:
        b = s.get('buoy') or {}
        if b.get('source') != 'NDBC':
            continue
        st, (y0, y1) = b['station'], b['years']
        frames = []
        for y in range(y0, y1 + 1):
            p = fetch(st, y)
            if p:
                frames.append(parse(p))
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated()]
        out = pd.DataFrame({'hs': df.get('WVHT'), 'tp': df.get('DPD'), 'tm02': df.get('APD'), 'mwd': df.get('MWD'),
                            'wspd': df.get('WSPD'), 'wdir': df.get('WDIR'), 'pres': df.get('PRES'),
                            'atmp': df.get('ATMP'), 'wtmp': df.get('WTMP')}, index=df.index)
        out = out[out['hs'].between(0, 25) | out['hs'].isna()]           # 去掉明显错误值
        out.index.name = 'valid_time'
        out.reset_index().assign(valid_time=lambda d: d['valid_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')) \
            .to_csv(os.path.join(OUT, f"{s['id']}_{st}.csv"), index=False)
        hs = out['hs'].dropna()
        print(f"[完成] {s['id']} NDBC {st}：{hs.index.min():%Y-%m-%d} – {hs.index.max():%Y-%m-%d}，"
              f"有效 Hs {len(hs)} 条，平均 {hs.mean():.2f} m，最大 {hs.max():.2f} m（{hs.idxmax():%Y-%m-%d}）")


if __name__ == '__main__':
    main()
