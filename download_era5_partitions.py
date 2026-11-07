"""ERA5 风浪 / 涌浪分量（单点时间序列数据集没有这些变量，所以用普通网格数据集按站点小范围下载）。

变量：风浪（本地风生成的浪）有效波高、平均周期、平均方向；总涌浪（远处传来的浪）有效波高、平均周期、平均方向。
为什么有用：台风到来之前，涌浪往往先到；把"浪里有多少是涌浪、从哪个方向来"作为过去 7 天的输入，可以提示远处的天气系统。
请求：每个站点每年两个请求（风浪 3 个变量、涌浪 3 个变量，逐 3 小时，约 8,760 个场，在 CDS 单次请求的费用上限内），
     范围是站点周围 ±0.5°（海浪网格 0.5°），数据量很小（每站每年约 1 MB 以内）。
输出：era5_data/partitions_<站点名>_<年>_<wind|swell>.nc
用法：python download_era5_partitions.py [G K H ...] [--years 2000 2024]     已存在的自动跳过；限流时自动重试
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

import cdsapi

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = {s['id']: s for s in json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']}
OUT = os.path.join(ROOT, 'era5_data')
LOG = os.path.join(OUT, 'download_log.txt')
GROUPS = {'wind': ['significant_height_of_wind_waves', 'mean_period_of_wind_waves', 'mean_direction_of_wind_waves'],
          'swell': ['significant_height_of_total_swell', 'mean_period_of_total_swell', 'mean_direction_of_total_swell']}
WORKERS = 2


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def job(sid, year, group):
    site = SITES[sid]
    lat, lon = site.get('ts_point') or site['point']
    target = os.path.join(OUT, f"partitions_{site['name']}_{year}_{group}.nc")
    if os.path.exists(target) and os.path.getsize(target) > 2000:
        return f'[跳过] {os.path.basename(target)}'
    req = {'product_type': ['reanalysis'], 'variable': GROUPS[group], 'year': [str(year)],
           'month': [f'{m:02d}' for m in range(1, 13)], 'day': [f'{d:02d}' for d in range(1, 32)],
           'time': [f'{h:02d}:00' for h in range(0, 24, 3)], 'area': [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5],
           'data_format': 'netcdf', 'download_format': 'unarchived'}
    client = cdsapi.Client(verify=0, quiet=True)
    for attempt in range(40):
        try:
            client.retrieve('reanalysis-era5-single-levels', req, target + '.part')
            os.replace(target + '.part', target)
            return f'[完成] {os.path.basename(target)} {os.path.getsize(target) / 1e3:.0f} KB'
        except Exception as e:
            msg = str(e).splitlines()[-1][:160]
            wait = 600 if ('limited' in msg or 'queued' in msg.lower() or 'Max retries' in msg) else 60 * min(attempt + 1, 10)
            log(f'[重试 {attempt + 1}] 分量 {sid} {year} {group}：{msg}')
            time.sleep(wait)
    return f'[失败] 分量 {sid} {year} {group}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sites', nargs='*', default=['G', 'K', 'H'])
    p.add_argument('--years', nargs=2, type=int, default=[2000, 2024])
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, y, g) for s in a.sites for y in range(a.years[0], a.years[1] + 1) for g in GROUPS]
    log(f'风浪/涌浪分量计划 {len(jobs)} 个请求：' + '、'.join(a.sites))
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for fut in cf.as_completed([ex.submit(job, *j) for j in jobs]):
            log(fut.result())
    log('=== 风浪/涌浪分量本轮结束（失败的重跑即可补齐）===')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    if sys.stdout is None:
        os.makedirs(OUT, exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(OUT, 'partitions_run_log.txt'), 'a', encoding='utf-8', buffering=1)
    main()
