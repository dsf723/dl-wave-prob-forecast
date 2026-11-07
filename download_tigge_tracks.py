"""TIGGE 热带气旋预报路径（NCAR GDEX 数据集 d330003，ECMWF，CXML 格式），经 OSDF 直链下载，不需要登录。

为什么选 ECMWF：每个文件同时含确定性预报 + 50 个集合成员的路径、中心气压、最大风速，文件只有 0.4–0.8 MB；
NCEP GEFS 集合路径每个文件约 35 MB（每天 4 次），太大。
只下 12 UTC 起报：GEFS 00 UTC 起报、决策时刻 06 UTC 时，ECMWF 当天 00 UTC 的产品还没发布（约 07–09 UTC），
能用的最新一次是前一天 12 UTC（当晚即发布），用它才不会"偷看"。2006-10（TIGGE 开始）至 2019-12（GEFS 回算结束）。
文件名：早期为 ..._<HH>0000_ifs_glob_test_all_glo.xml，之后为 ..._ifs_glob_prod_all_glo.xml，两种都试。
输出：tigge_tracks/ecmf/<年>/<文件名>.xml；下载记录 tigge_tracks/download_log.txt（缺失的日期也记下）
用法：python download_tigge_tracks.py [--start 2006-10-01] [--end 2019-12-31] [--hour 12] [--workers 4]   已存在的自动跳过
"""
import argparse
import concurrent.futures as cf
import os
import sys
import time

import pandas as pd
import requests
import urllib3

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'tigge_tracks', 'ecmf')
LOG = os.path.join(ROOT, 'tigge_tracks', 'download_log.txt')
BASE = 'https://osdf-director.osg-htc.org/ncar/gdex/d330003/ecmf'
urllib3.disable_warnings()


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def job(day, hour):
    """当天 hour 时的文件；12 UTC 缺失时退而用同一天 00 UTC（更早发布，同样不偷看）。"""
    d = day.strftime('%Y%m%d')
    folder = os.path.join(OUT, d[:4])
    hours = [hour] + ([0] if hour == 12 else [])
    names = [f'z_tigge_c_ecmf_{d}{h:02d}0000_ifs_glob_{kind}_all_glo.xml' for h in hours for kind in ('prod', 'test')]
    if any(os.path.exists(os.path.join(folder, n)) for n in names):
        return 'skip', 0
    for attempt in range(5):
        try:
            for name in names:
                r = requests.get(f'{BASE}/{d[:4]}/{d}/{name}', timeout=120, verify=False)
                if r.status_code == 200 and r.content.lstrip().startswith(b'<?xml'):
                    os.makedirs(folder, exist_ok=True)
                    tmp = os.path.join(folder, name + '.part')
                    open(tmp, 'wb').write(r.content)
                    os.replace(tmp, os.path.join(folder, name))
                    return ('ok' if f'{d}{hour:02d}0000' in name else 'fallback'), len(r.content)
                if r.status_code not in (404, 403):
                    raise RuntimeError(f'HTTP {r.status_code}')
            return 'missing', 0
        except Exception as e:
            err = str(e)[:120]
            time.sleep(10 * (attempt + 1))
    log(f'[失败] {d}：{err}')
    return 'fail', 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2006-10-01')
    p.add_argument('--end', default='2019-12-31')
    p.add_argument('--hour', type=int, default=12, choices=[0, 12])
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    days = pd.date_range(a.start, a.end, freq='D')
    log(f'TIGGE ECMWF 路径：{a.start} 至 {a.end}，共 {len(days)} 天（{a.hour:02d} UTC）')
    count = {'ok': 0, 'fallback': 0, 'skip': 0, 'missing': 0, 'fail': 0}
    total = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(job, d, a.hour): d for d in days}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            status, size = fut.result()
            count[status] += 1
            total += size
            if status == 'fallback':
                count['ok'] += 1
            if status == 'missing':
                log(f'[缺失] {futs[fut]:%Y-%m-%d} 没有 {a.hour:02d} UTC 文件（也没有 00 UTC 可替代）')
            if i % 100 == 0 or i == len(days):
                log(f'进度 {i}/{len(days)}：新下 {count["ok"]}（其中改用 00 UTC {count["fallback"]}）、已有 {count["skip"]}、缺失 {count["missing"]}、失败 {count["fail"]}，'
                    f'本次 {total / 1e6:.0f} MB，{(time.time() - t0) / 60:.1f} 分钟')
    log('=== TIGGE 路径下载本轮结束（失败的重跑即可补齐）===')


if __name__ == '__main__':
    if sys.stdout is None:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(os.path.dirname(LOG), 'run_log.txt'), 'a', encoding='utf-8', buffering=1)
    else:
        sys.stdout.reconfigure(encoding='utf-8')
    main()
