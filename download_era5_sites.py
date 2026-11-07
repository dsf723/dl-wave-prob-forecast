"""按站点下载 ERA5 逐时网格资料（1995–2024），站点见 metadata/sites.json。
【备用方案】主方案是 download_era5_timeseries.py（单点时间序列，请求少得多）。本脚本按季度 × 变量组请求，
请求数多、排队慢，用于：时间序列数据集长期被限流时；或需要谱峰周期 pp1d、跨零周期 mp2（时间序列数据集没有）时，
例如：python download_era5_sites.py G H   （浮标站的周期对比）。

与 download_era5_full.py 的区别：
- 站点改为 A 舟山、B 珠海、C 平潭、G 关岛、K 济州南、H 夏威夷 WETS（去掉俄勒冈）；
- 变量增加 谱峰周期 pp1d、跨零周期 mp2（与浮标的 DPD、APD 对应）、2 m 气温、海平面气压（判定寒潮用）；
- 按年请求（请求数是按季度的 1/4，排队时间少很多）；请求过大时自动拆成上下半年；
- 舟山已按季度下载了 5 个变量的年份，只补下缺少的 4 个变量（*_extra.nc），不重复下载。

用法：
  python download_era5_sites.py              # 全部站点
  python download_era5_sites.py B C          # 只下指定站点
  python download_era5_sites.py --a-full     # 舟山缺季度的年份也整年下载（旧脚本停止后再用）
  python download_era5_sites.py G H --periods   # 只下浮标年份的 pp1d、mp2（按半年，文件 *_H1_per.nc），供周期对比
已存在且大于 10 KB 的文件自动跳过，可随时中断后重跑。
"""
import concurrent.futures as cf
import json
import os
import sys
import time

import cdsapi

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
OUT = os.path.join(ROOT, 'era5_data')
LOG = os.path.join(OUT, 'download_log.txt')
YEARS = list(range(1995, 2025))
WORKERS = 2                                   # 同时排队的请求数；旧脚本若仍在运行，它也占一个

VARS_OLD = ['significant_height_of_combined_wind_waves_and_swell', 'mean_wave_period', 'mean_wave_direction',
            '10m_u_component_of_wind', '10m_v_component_of_wind']
VARS_EXTRA = ['peak_wave_period', 'mean_zero_crossing_wave_period', '2m_temperature', 'mean_sea_level_pressure']
VARS_FULL = VARS_OLD + VARS_EXTRA
HOURS = [f'{h:02d}:00' for h in range(24)]


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def ok(path):
    return os.path.exists(path) and os.path.getsize(path) > 10000


def retrieve(client, area, variables, year, months, target):
    req = {'product_type': ['reanalysis'], 'variable': variables, 'year': [str(year)], 'month': months,
           'day': [f'{d:02d}' for d in range(1, 32)], 'time': HOURS, 'area': area,
           'data_format': 'netcdf', 'download_format': 'unarchived'}
    client.retrieve('reanalysis-era5-single-levels', req, target + '.part')
    os.replace(target + '.part', target)


QUARTERS = {'Q1': ['01', '02', '03'], 'Q2': ['04', '05', '06'], 'Q3': ['07', '08', '09'], 'Q4': ['10', '11', '12']}
WAVE = {'significant_height_of_combined_wind_waves_and_swell', 'mean_wave_period', 'mean_wave_direction',
        'peak_wave_period', 'mean_zero_crossing_wave_period'}


def job(site, year, variables, tag):
    """下载一个站点一年。CDS 对单次请求有费用上限（实测：5 个变量 × 一季度可以，9 个变量 × 半年超限），
    因此按季度、并把海浪变量和大气变量分成两组请求：文件名 <站点>_<年>_<季度>_<wave|atm><tag>.nc。"""
    client = cdsapi.Client(verify=0, quiet=True)
    name = site['name']
    groups = [('wave', [v for v in variables if v in WAVE]), ('atm', [v for v in variables if v not in WAVE])]
    done = 0
    for q, months in QUARTERS.items():
        for gname, gvars in groups:
            if not gvars:
                continue
            target = os.path.join(OUT, f'{name}_{year}_{q}_{gname}{tag}.nc')
            if ok(target):
                done += 1
                continue
            for attempt in range(6):
                try:
                    retrieve(client, site['area'], gvars, year, months, target)
                    done += 1
                    break
                except Exception as e:
                    msg = str(e).splitlines()[-1][:160]
                    wait = 600 if 'limited' in msg or 'queued' in msg.lower() else 60 * (attempt + 1)
                    log(f'[重试 {attempt + 1}] {name} {year} {q} {gname}{tag}：{msg}')
                    time.sleep(wait)
    return f'[完成] {name} {year}{tag}：{done} 个文件就绪'


VARS_PERIOD = ['peak_wave_period', 'mean_zero_crossing_wave_period']
HALVES = {'H1': [f'{m:02d}' for m in range(1, 7)], 'H2': [f'{m:02d}' for m in range(7, 13)]}


def job_periods(site, year):
    """只下谱峰周期、跨零周期两个变量（时间序列数据集没有），2 个变量 × 半年在费用上限内。"""
    client = cdsapi.Client(verify=0, quiet=True)
    done = 0
    for h, months in HALVES.items():
        target = os.path.join(OUT, f"{site['name']}_{year}_{h}_per.nc")
        if ok(target):
            done += 1
            continue
        for attempt in range(40):                      # 网络中断可能持续数小时（10-16 凌晨约 9 小时），最长等约 6 小时
            try:
                retrieve(client, site['area'], VARS_PERIOD, year, months, target)
                done += 1
                break
            except Exception as e:
                msg = str(e).splitlines()[-1][:160]
                wait = 600 if ('limited' in msg or 'queued' in msg.lower() or 'Max retries' in msg) else 60 * min(attempt + 1, 10)
                log(f"[重试 {attempt + 1}] {site['name']} {year} {h} 周期：{msg}")
                time.sleep(wait)
    return f"[完成] {site['name']} {year} 周期：{done}/2 个文件就绪"


def plan_periods(selected):
    """浮标站在浮标有资料的年份（与 1995–2024 的交集）下载 pp1d、mp2。"""
    jobs = []
    for site in sorted(SITES, key=lambda s: s['priority']):
        years = (site.get('buoy') or {}).get('years')
        if (selected and site['id'] not in selected) or not years:
            continue
        jobs += [(site, y) for y in YEARS if years[0] <= y <= years[1]]
    return jobs


def plan(selected, a_full):
    jobs = []
    for site in sorted(SITES, key=lambda s: s['priority']):
        if selected and site['id'] not in selected:
            continue
        for year in YEARS:
            quarters = [os.path.join(OUT, f'{site["name"]}_{year}_Q{q}.nc') for q in range(1, 5)]
            if site['name'] == 'zhoushan' and all(ok(q) for q in quarters):
                jobs.append((site, year, VARS_EXTRA, '_extra'))       # 旧脚本已下 5 个变量，只补 4 个
            elif site['name'] == 'zhoushan' and not a_full:
                continue                                              # 等旧脚本下完，或加 --a-full
            else:
                jobs.append((site, year, VARS_FULL, ''))
    return jobs


def main():
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:]
    a_full = '--a-full' in args
    selected = {a for a in args if not a.startswith('--')}
    if '--periods' in args:
        jobs = plan_periods(selected)
        log(f'计划 {len(jobs)} 个站点年（周期变量）：' + '、'.join(sorted({j[0]['id'] for j in jobs})))
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            for fut in cf.as_completed([ex.submit(job_periods, *j) for j in jobs]):
                log(fut.result())
        log('=== 周期变量本轮结束（失败的请求重跑本脚本即可补齐）===')
        return
    jobs = plan(selected, a_full)
    log(f'计划 {len(jobs)} 个请求：' + '、'.join(sorted({j[0]['id'] for j in jobs})))
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for fut in cf.as_completed([ex.submit(job, *j) for j in jobs]):
            log(fut.result())
    log('=== 本轮结束（失败的请求重跑本脚本即可补齐）===')


if __name__ == '__main__':
    if sys.stdout is None:                    # pythonw（无控制台）运行时：输出写入日志文件
        os.makedirs(OUT, exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(OUT, 'sites_run_log.txt'), 'a', encoding='utf-8', buffering=1)
    main()
