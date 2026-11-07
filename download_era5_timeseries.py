"""用 CDS 的 ERA5 单点时间序列数据集（reanalysis-era5-single-levels-timeseries）下载各站点 1995–2024 逐时资料。

为什么用它：普通 ERA5（reanalysis-era5-single-levels）对单次请求有费用上限，9 个变量连半年都超限，
按季度拆分要排上千个请求；单点时间序列数据集专为提取一个点的长序列设计，一个站点一次（或分几段）即可。
限制：取最近的格点（海浪 0.5°、大气 0.25°），不插值；该数据集没有谱峰周期 pp1d 和跨零周期 mp2。

站点与坐标见 metadata/sites.json；若站点设了 ts_point（取时间序列用的点，如岛屿浮标附近的海上点），优先用它。
输出：era5_data/ts_<站点名>_<起年>_<止年>.nc（若返回 zip 也保留原样，转换脚本会自动识别）。
用法：python download_era5_timeseries.py [B C G ...] [--long]     已存在的文件自动跳过；被 CDS 限流时会等待后重试。
  --long：再下 1940–1994（分三段），用于更长的气候态（10-22 加）；先下完所有站点的 1995–2024，再下长序列。
"""
import json
import os
import sys
import time

import cdsapi

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
OUT = os.path.join(ROOT, 'era5_data')
LOG = os.path.join(OUT, 'download_log.txt')
VARS = ['significant_height_of_combined_wind_waves_and_swell', 'mean_wave_period', 'mean_wave_direction',
        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure']
PERIODS = [(1995, 2009), (2010, 2024)]          # 分两段，降低单次请求被拒的可能
LONG_PERIODS = [(1940, 1959), (1960, 1979), (1980, 1994)]


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def get(client, site, y0, y1):
    lat, lon = site.get('ts_point') or site['point']
    target = os.path.join(OUT, f"ts_{site['name']}_{y0}_{y1}.nc")
    if os.path.exists(target) and os.path.getsize(target) > 10000:
        log(f'[跳过] {os.path.basename(target)}')
        return
    req = {'variable': VARS, 'location': {'longitude': lon, 'latitude': lat},
           'date': [f'{y0}-01-01/{y1}-12-31'], 'data_format': 'netcdf'}
    for attempt in range(72):                    # 被限流时每 10 分钟重试，最多约 12 小时
        try:
            t = time.time()
            client.retrieve('reanalysis-era5-single-levels-timeseries', req, target + '.part')
            os.replace(target + '.part', target)
            log(f'[完成] {os.path.basename(target)} {os.path.getsize(target) / 1e6:.1f} MB，用时 {time.time() - t:.0f} s')
            return
        except Exception as e:
            msg = str(e)
            wait = 600 if 'temporarily limited' in msg or 'queued' in msg.lower() else 120
            log(f'[等待重试 {attempt + 1}] {site["id"]} {y0}–{y1}：{msg.splitlines()[-1][:160]}；{wait // 60} 分钟后重试')
            time.sleep(wait)
    log(f'[失败] {site["id"]} {y0}–{y1}')


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = {a for a in sys.argv[1:] if not a.startswith('--')}
    passes = [PERIODS] + ([LONG_PERIODS] if '--long' in sys.argv else [])
    client = cdsapi.Client(verify=0, quiet=True)   # 与原脚本一致：本机经代理访问，关闭证书校验
    for periods in passes:
        for site in sorted(SITES, key=lambda s: s['priority']):
            if sel and site['id'] not in sel:
                continue
            for y0, y1 in periods:
                get(client, site, y0, y1)
    log('=== 时间序列下载本轮结束 ===')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    if sys.stdout is None:                    # pythonw（无控制台）运行时：输出写入日志文件
        os.makedirs(OUT, exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(OUT, 'timeseries_run_log.txt'), 'a', encoding='utf-8', buffering=1)
    main()
