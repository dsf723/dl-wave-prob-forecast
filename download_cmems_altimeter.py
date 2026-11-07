"""CMEMS 卫星高度计沿轨有效波高（L3 再处理，ESA CCI，多卫星合并，1 Hz）在各站点附近的提取。

产品：WAVE_GLO_PHY_SWH_L3_MY_014_005，数据集 cci_obs-wave_glo_phy-swh_my_l3_PT1S-i（1991-08 至 2023-12）。
变量：swh_adjusted（相互校准后的 Hs，主用）、swh、swh_denoised、swh_uncertainty。
范围：以 sites.json 的 point 为中心，纬向 ±BOX_DEG，经向 ±BOX_DEG/cos(纬度)（约 ±55 km 方框）；按年请求。
输出：raw/altimeter/<站点ID>/<年>.csv（工具箱稀疏子集输出），合并后 processed/altimeter/<站点ID>_cci_l3.csv。
登录：需先由本人在终端运行 `copernicusmarine login`（凭据由 copernicusmarine 保存在本地，脚本不接触密码）。
用法：python download_cmems_altimeter.py [A B C ...]   已存在的年份自动跳过
"""
import glob
import json
import math
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
DATASET = 'cci_obs-wave_glo_phy-swh_my_l3_PT1S-i'
VARS = ['swh_adjusted', 'swh', 'swh_denoised', 'swh_uncertainty']
YEARS = range(1995, 2024)
BOX_DEG = 0.5
KEEP = ['variable', 'platform_id', 'time', 'longitude', 'latitude', 'value', 'value_qc']
RAW = os.path.join(ROOT, 'raw', 'altimeter')
OUT = os.path.join(ROOT, 'processed', 'altimeter')


def get_year(cm, site, year):
    lat, lon = site['point']
    dlon = BOX_DEG / math.cos(math.radians(lat))
    d = os.path.join(RAW, site['id'])
    os.makedirs(d, exist_ok=True)
    target = os.path.join(d, f'{year}.csv')
    if os.path.exists(target):
        return 'skip'
    for k in range(4):
        try:
            cm.subset(dataset_id=DATASET, variables=VARS,
                      minimum_latitude=lat - BOX_DEG, maximum_latitude=lat + BOX_DEG,
                      minimum_longitude=lon - dlon, maximum_longitude=lon + dlon,
                      start_datetime=f'{year}-01-01T00:00:00', end_datetime=f'{year}-12-31T23:59:59',
                      output_directory=d, output_filename=f'{year}.csv', file_format='csv',
                      disable_progress_bar=True)
            if not os.path.exists(target):              # 当年无过境：写空文件占位，避免重复请求
                open(target, 'w').close()
            else:                                       # 工具箱输出为长表且含机构全称，只留需要的列（约缩小 5 倍）
                df = pd.read_csv(target)
                df[KEEP].to_csv(target, index=False)
            return 'ok'
        except Exception as e:
            msg = str(e)
            if 'credentials' in msg.lower() or 'login' in msg.lower():
                raise SystemExit('未找到 Copernicus Marine 登录凭据：请先在终端运行 copernicusmarine login')
            print(f'  [重试 {k + 1}] {site["id"]} {year}：{msg[:200]}', flush=True)
            time.sleep(30 * (k + 1))
    return 'fail'


def merge(site):
    parts = []
    for f in sorted(glob.glob(os.path.join(RAW, site['id'], '*.csv'))):
        if os.path.getsize(f) == 0:
            continue
        parts.append(pd.read_csv(f))
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    df = df.pivot_table(index=['time', 'platform_id', 'longitude', 'latitude'], columns='variable', values='value').reset_index()
    df.columns.name = None
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{site['id']}_cci_l3.csv")
    df.to_csv(p, index=False)
    print(f"[合并] {site['id']}：{len(df)} 行 → {p}；列 {list(df.columns)}")


def main():
    import copernicusmarine as cm
    sel = set(sys.argv[1:])
    for site in sorted(SITES, key=lambda s: s['priority']):
        if sel and site['id'] not in sel:
            continue
        t0 = time.time()
        res = [get_year(cm, site, y) for y in YEARS]
        print(f"[{site['id']}] {site['name']}：完成 {res.count('ok')}，跳过 {res.count('skip')}，失败 {res.count('fail')}，"
              f"用时 {time.time() - t0:.0f} s", flush=True)
        merge(site)


if __name__ == '__main__':
    if sys.stdout is None:                    # pythonw（无控制台）运行时：输出写入日志文件
        os.makedirs(RAW, exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(RAW, '_run_log.txt'), 'a', encoding='utf-8', buffering=1)
    else:
        sys.stdout.reconfigure(encoding='utf-8')
    main()
