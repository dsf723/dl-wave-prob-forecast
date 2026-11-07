"""韩国气象厅（KMA）海洋气象浮标逐时资料 → processed/buoy/<站点ID>_<站号>.csv（与 NDBC 浮标同一格式）。

资料来源：data.kma.go.kr → 데이터 → 기상관측 → 해양 → 해양기상부이 → 파일셋，자료형태 = 시간 자료，
逐年文件 MARINE_BUOY_<站号>_HR_<年>_*.zip（需登录下载）；下载的打包文件放在 raw/observations/kma/，
解压后的 CSV 在 raw/observations/kma/csv/（cp949 编码）。
处理：
  - 时间：KMA 为韩国标准时（KST = UTC+9），转为 UTC，写成 2008-01-01T00:00:00Z；
  - 列：유의파고 → hs，파주기 → tp_kma（KMA 未说明是谱峰还是平均周期，暂不与 tp/tm02 混用），파향 → mwd，
    풍속/풍향 → wspd/wdir，현지기압 → pres，기온 → atmp，수온 → wtmp，최대파고 → hmax；
  - 质控：hs 限 0–25 m，且不超过 최대파고；风速 0–80 m/s；气压 900–1100 hPa；重复时刻取第一条。
台风标签：运行后再跑 python label_typhoons.py，会回填 label_tc_distance_km、event_id。
用法：python prepare_kma_buoy.py            （处理 sites.json 中 buoy.source == 'KMA' 的站点）
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
RAW = os.path.join(ROOT, 'raw', 'observations', 'kma', 'csv')
OUT = os.path.join(ROOT, 'processed', 'buoy')
COLS = {'유의파고(m)': 'hs', '파주기(sec)': 'tp_kma', '파향(deg)': 'mwd', '풍속(m/s)': 'wspd', '풍향(deg)': 'wdir',
        '현지기압(hPa)': 'pres', '기온(°C)': 'atmp', '수온(°C)': 'wtmp', '최대파고(m)': 'hmax'}
LIMITS = {'hs': (0, 25), 'hmax': (0, 40), 'tp_kma': (1, 30), 'mwd': (0, 360), 'wspd': (0, 80), 'wdir': (0, 360),
          'pres': (900, 1100), 'atmp': (-30, 45), 'wtmp': (-2, 35)}


def convert(site):
    st = site['buoy']['station']
    files = sorted(glob.glob(os.path.join(RAW, f'MARINE_BUOY_{st}_HR_*.csv')))
    if not files:
        print(f"[跳过] {site['id']}：{RAW} 下没有 {st} 的逐时文件")
        return
    df = pd.concat(pd.read_csv(f, encoding='cp949') for f in files)
    t = pd.to_datetime(df['일시']).dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC').dt.tz_localize(None)
    out = pd.DataFrame({c: pd.to_numeric(df[k], errors='coerce').values for k, c in COLS.items() if k in df},
                       index=pd.DatetimeIndex(t, name='valid_time'))
    out = out[~out.index.duplicated()].sort_index()
    for c, (lo, hi) in LIMITS.items():
        if c in out:
            out.loc[(out[c] < lo) | (out[c] > hi), c] = np.nan
    if 'hmax' in out:
        out.loc[out['hs'] > out['hmax'], 'hs'] = np.nan
    for c in ('tp', 'tm02'):                     # 与 NDBC 文件列一致，KMA 无对应定义明确的周期
        out[c] = np.nan
    out['label_tc_distance_km'] = np.nan
    out['event_id'] = ''
    order = ['hs', 'tp', 'tm02', 'mwd', 'wspd', 'wdir', 'pres', 'atmp', 'wtmp', 'hmax', 'tp_kma', 'label_tc_distance_km', 'event_id']
    out = out[order].reset_index()
    out['valid_time'] = out['valid_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{site['id']}_{st}.csv")
    out.to_csv(p, index=False)
    hs = out['hs'].dropna()
    print(f"[完成] {site['id']} KMA {st}：{len(files)} 个年文件，{out.valid_time.iloc[0]} – {out.valid_time.iloc[-1]}，"
          f"{len(out)} 行，Hs 有效 {len(hs)}（{len(hs) / len(out):.0%}），均值 {hs.mean():.2f} m，最大 {hs.max():.1f} m → {p}")


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    for s in SITES:
        if (s.get('buoy') or {}).get('source') == 'KMA':
            convert(s)
