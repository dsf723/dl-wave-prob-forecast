"""把 TIGGE ECMWF 路径 XML（download_tigge_tracks.py 下载的）压缩成一张表，只保留站点所在海域的热带气旋。

保留范围：纬度 -5°～55°N，经度 95°E～110°W（含西北太平洋 G / K 站、中太平洋与东太平洋 H 站）。
每行 = 一个起报时刻 × 一个成员 × 一个气旋 × 一个预报时效：
  base_time、member（-1 = 确定性预报，0 = 控制预报，1–50 = 集合成员）、storm_id、basin、lead_h、valid_time、
  lat（°N，南纬为负）、lon（0–360°E）、pmin_hpa（中心气压）、vmax_ms（最大风速）
输出：tigge_tracks/ecmf_tracks_pacific.csv.gz（几十 MB，服务器上解析后只需拷回这一个文件）
用法：python parse_tigge_tracks.py [--src tigge_tracks/ecmf]
"""
import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
LAT, LON = (-5, 55), (95, 250)


def signed(node, tag):
    e = node.find(tag)
    if e is None or not (e.text or '').strip():
        return None
    v = float(e.text)
    unit = e.get('units', '')
    if 'S' in unit or 'W' in unit:
        v = -v
    return v


def parse(path):
    rows = []
    root = ET.parse(path).getroot()
    base = root.findtext('header/baseTime')
    for data in root.iter('data'):
        kind = data.get('type')
        if kind == 'forecast':
            member = -1
        elif kind == 'ensembleForecast':
            member = int(data.get('member', 0))
        else:
            continue
        for dist in data.iter('disturbance'):
            sid, basin = dist.get('ID'), dist.findtext('basin')
            for fix in dist.iter('fix'):
                lat, lon = signed(fix, 'latitude'), signed(fix, 'longitude')
                if lat is None or lon is None:
                    continue
                lon %= 360
                if not (LAT[0] <= lat <= LAT[1] and LON[0] <= lon <= LON[1]):
                    continue
                p = fix.findtext('cycloneData/minimumPressure/pressure')
                w = fix.findtext('cycloneData/maximumWind/speed')
                rows.append((base, member, sid, basin, int(fix.get('hour', 0)), fix.findtext('validTime'), lat, lon,
                             float(p) if p else None, float(w) if w else None))
    return rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=os.path.join(ROOT, 'tigge_tracks', 'ecmf'))
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, '*', '*.xml')))
    rows, bad = [], []
    for i, f in enumerate(files, 1):
        try:
            rows += parse(f)
        except ET.ParseError as e:
            bad.append(f'{os.path.basename(f)}：{e}')
        if i % 500 == 0:
            print(f'已解析 {i}/{len(files)} 个文件，{len(rows)} 行', flush=True)
    df = pd.DataFrame(rows, columns=['base_time', 'member', 'storm_id', 'basin', 'lead_h', 'valid_time',
                                     'lat', 'lon', 'pmin_hpa', 'vmax_ms'])
    out = os.path.join(ROOT, 'tigge_tracks', 'ecmf_tracks_pacific.csv.gz')
    df.to_csv(out, index=False, compression='gzip')
    print(f'{len(files)} 个文件 → {len(df)} 行，{df.base_time.nunique()} 次起报，{df.storm_id.nunique()} 个气旋编号 → {out}')
    if bad:
        print(f'{len(bad)} 个文件解析失败：\n' + '\n'.join(bad[:20]))


if __name__ == '__main__':
    main()
