"""ERA5 区域 10 m 风场（1° 网格、逐 3 小时、1995–2024），用于计算站点上游扇区风特征（wind_features.py）。

为什么要风场：站点的浪 = 本地风浪 + 远处传来的涌浪；涌浪由几百到上千公里外的风生成，只看站点处的风看不到。
区域：BOXES 中每个框各一组文件。东亚框（10–44°N、103–137°E）覆盖 A 舟山、B 珠海、C 平潭、K 马罗岛周围约 1000 km；
     关岛、夏威夷各一个框（需要时加 G、H 参数）。
请求：每个框每年一个请求（2 个变量 × 8 个时次 × 365 天 ≈ 5,800 个场，在 CDS 单次请求费用上限内），
     用 grid=1.0 在服务器端插值到 1°，数据量约 30 MB/年/框。
输出：era5_data/windfield_<框名>_<年>.nc
用法：python download_era5_windfield.py [EA G H]     默认只下东亚框 EA；已存在的年份自动跳过
"""
import concurrent.futures as cf
import os
import sys
import time

import cdsapi

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'era5_data')
LOG = os.path.join(OUT, 'download_log.txt')
YEARS = range(1995, 2025)
BOXES = {'EA': [44, 103, 10, 137],             # 北、西、南、东
         'G': [24, 134, 3, 155],
         'H': [32, -168, 11, -147]}
WORKERS = 2


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def job(box, year):
    target = os.path.join(OUT, f'windfield_{box}_{year}.nc')
    if os.path.exists(target) and os.path.getsize(target) > 10000:
        return f'[跳过] {os.path.basename(target)}'
    client = cdsapi.Client(verify=0, quiet=True)
    req = {'product_type': ['reanalysis'], 'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
           'year': [str(year)], 'month': [f'{m:02d}' for m in range(1, 13)], 'day': [f'{d:02d}' for d in range(1, 32)],
           'time': [f'{h:02d}:00' for h in range(0, 24, 3)], 'area': BOXES[box], 'grid': [1.0, 1.0],
           'data_format': 'netcdf', 'download_format': 'unarchived'}
    for attempt in range(40):                  # 限流或断网时最多等约 6 小时
        try:
            t = time.time()
            client.retrieve('reanalysis-era5-single-levels', req, target + '.part')
            os.replace(target + '.part', target)
            return f'[完成] {os.path.basename(target)} {os.path.getsize(target) / 1e6:.1f} MB，用时 {time.time() - t:.0f} s'
        except Exception as e:
            msg = str(e).splitlines()[-1][:160]
            wait = 600 if ('limited' in msg or 'queued' in msg.lower() or 'Max retries' in msg) else 60 * min(attempt + 1, 10)
            log(f'[重试 {attempt + 1}] 风场 {box} {year}：{msg}')
            time.sleep(wait)
    return f'[失败] 风场 {box} {year}'


def main():
    os.makedirs(OUT, exist_ok=True)
    boxes = [a for a in sys.argv[1:] if a in BOXES] or ['EA']
    jobs = [(b, y) for b in boxes for y in YEARS]
    log(f'风场计划 {len(jobs)} 个请求：' + '、'.join(boxes))
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for fut in cf.as_completed([ex.submit(job, *j) for j in jobs]):
            log(fut.result())
    log('=== 风场本轮结束（失败的重跑本脚本即可补齐）===')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    if sys.stdout is None:
        os.makedirs(OUT, exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(OUT, 'windfield_run_log.txt'), 'a', encoding='utf-8', buffering=1)
    main()
