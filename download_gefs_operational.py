"""NOAA GEFSv12 业务海浪集合预报（2020-09-23 起，WAVEWATCH III）在浮标站上的提取 —— 独立业务检验用（10-27）。

为什么：2017–2019 测试集已用于 14 天模型的评价；GEFSv12 回算只到 2019 年，2020-09-23 起的业务预报是模型
都没见过的全新数据，用冻结的最终模型在上面评分即"独立业务检验"。
资料：AWS 公开桶 noaa-gefs-pds（不用注册），
  gefs.<日期>/00/wave/gridded/gefs.wave.t00z.<成员>.global.0p25.f<时效>.grib2（+ .idx 索引）
  - 31 个成员（c00、p01–p30）+ mean / spread / prob；0.25° 全球；f000–f240 每 3 h、f246–f384 每 6 h。
  - 业务的 wave/station/ 只有集合平均统计、没有成员，且不含韩国 22107 → 成员只能从格点取。
做法：
  - 按 .idx 只读需要的 GRIB 记录（HTTP Range）：每个成员的 HTSGW（Hs，约 0.41 MB / 条）、MWSPER（平均周期，约 0.49 MB）、
    WWSDIR（合成波浪平均方向，约 0.78 MB）。mean 文件里没有这两个周期、方向变量（10-27 核实），所以要逐成员取。
  - **业务 GRIB 的周期与回算 tr 不同义**（MWSPER = GRIB 参数 10-0-15，ecCodes 名 mwp；IMWF = 10-0-25，mp1）。10-27 本机核对（c00，时效 6–24 h，2020-10 至 2024-12 每 28 天）：
    MWSPER ≈ 浮标 Tm02（比值 0.99）、与 ERA5 Te 相关 0.84–0.95（不低于回算 tr），但数值是 Te 的 0.81–0.87 倍；
    IMWF 更短（0.69–0.76 倍）、相关更低。→ 取 MWSPER 存为 tm02 列，tr 列留空；prepare_reforecast_dataset.py
    以 ERA5 为桥梁按站换算成 Te（系数约 1.17–1.22，见 metadata/gefs_reforecast_era5_ratios.json）。用回算 2019 年核对过：只用 c00 时
    第 8–15 天 Te 与 11 成员平均差 0.8–1.2 s、波向差 20–50°，与成员离散度相当 → 默认 11 个成员全取，与回算一致。
    默认取 c00、p01–p10 共 11 个成员，对应回算周三的 11 个成员；默认只取周三起报（与 GKH_h15 一致）。
  - 用 eccodes 解码全球场，按浮标经纬度做双线性插值；角点有陆地（缺测）时只用海上角点重新归一化，
    四角全缺测时取 5×5 窗口内最近的海上格点（每站用了哪种方式记在 _points.json）。
  - 每个成员的 Hs 按时效线性插值到逐 3 小时（f246 以后原生 6 小时，插值的行 native = 0）；波向按 sin / cos 插值。
  - 只用 GRIB 在内存里解码，不保存原始文件，磁盘占用很小；代价是流量。
输出：processed/gefs_operational/<站号>/<年>.csv.gz，列 valid_time, station, issue_time, member, hs, tr（空）, th1m, tm02, lat, lon,
      native（与回算 csv 同名列，prepare_reforecast_dataset.py 可直接读）；_done.txt 记已完成的起报日，
      _missing.txt 记存档缺失的起报日，_points.json 记各站插值方式。中断后用同一条命令重跑即可续传。
流量（周三、11 成员、到 f384、三个变量）：每个起报日约 1.9 GB，2020-09-23 至 2025-12-31 共约 275 次、约 530 GB；
  只到 2024-12-31（本机 ERA5 站点序列止于 2024 年底）约 224 次、约 430 GB。省流量：--member-vars HTSGW 只取波高（约 0.47 GB / 日），
  周期、波向就没有了，只能作 Hs 的敏感性试验。
用法（没有 eccodes 时 pip install eccodes）：
  python download_gefs_operational.py                              # 默认 G K H，2020-09-23 至 2024-12-31 的周三（约 224 次）
  python download_gefs_operational.py --end 2025-12-31             # ERA5 站点序列补到 2025 年后再续下 2025 年（已下的会跳过）
  python download_gefs_operational.py --sites G K H KR22101 ...    # 加站不加流量（同一张全球场多取几个点），--sites all = sites.json 里全部浮标
  python download_gefs_operational.py --start 2020-09-23 --end 2024-12-31 --procs 6 --threads 12
  bash run_oper_download.sh                                        # 按年份开多个 tmux 窗口并行（每窗口只写自己年份的文件）
  python download_gefs_operational.py --mirror gcs                 # 先取 Google Cloud 镜像（缺的退回 AWS）
  python download_gefs_operational.py --test                       # 冒烟：1 个起报日、2 个成员、前 24 小时，写到 processed/gefs_operational_test
  python download_gefs_operational.py --check-station 12           # 格点插值 vs 业务站点序列（集合平均 Hs，G、H）：抽 12 个起报日
然后整理：python prepare_reforecast_dataset.py G K H --sources gefs_operational --suffix _oper
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import io
import json
import os
import sys
import tarfile
import threading
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
BUCKET = 'https://noaa-gefs-pds.s3.amazonaws.com/'
# 10-27：Google Cloud 上有同样的存档（NOAA 开放数据计划），约 2020-10 起（2020-09-23 那天没有）；--mirror gcs 时先取镜像，
# 镜像没有的文件自动退回 AWS。哪个快取决于服务器所在网络，先用 curl 各下一个文件比较。
MIRRORS = {'aws': BUCKET, 'gcs': 'https://storage.googleapis.com/gfs-ensemble-forecast-system/'}
FIRST_DAY = '2020-09-23'                 # 业务海浪集合进入 GEFS 存档的第一天（10-27 列目录核实）
SITES = {s['id']: s for s in json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']}
MEMBERS = ['c00'] + [f'p{i:02d}' for i in range(1, 11)]
FHOURS = list(range(3, 241, 3)) + list(range(246, 385, 6))
COL = {'HTSGW': 'hs', 'MWSPER': 'tm02', 'IMWF': 'imwf', 'WWSDIR': 'th1m'}
DIRECTIONAL = {'WWSDIR'}
PARAM = {'HTSGW': (10, 0, 3), 'IMWF': (10, 0, 25), 'MWSPER': (10, 0, 15), 'PERPW': (10, 0, 11), 'WWSDIR': (10, 0, 14),
         'DIRPW': (10, 0, 10)}          # GRIB2 学科-类别-编号（10-27 对照 .idx 核实），无索引时按它找记录
_GRIB_LOCK = threading.Lock()              # ecCodes 多线程同时初始化会报 boot.def 找不到（10-27 本机复现），创建消息时加锁


def url_of(day, member, fh, base=BUCKET):
    return f'{base}gefs.{day}/00/wave/gridded/gefs.wave.t00z.{member}.global.0p25.f{fh:03d}.grib2'


class Missing(Exception):
    """存档里没有这个文件（404）。"""


def fetch(ses, url, rng=None, tries=8):
    """GET（可带字节段），网络错误按 15 s、30 s … 最长 10 分钟退避重试；404 直接报 Missing。"""
    for k in range(tries):
        try:
            r = ses.get(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}'} if rng else {}, timeout=120)
            if r.status_code == 404:
                raise Missing(url)
            r.raise_for_status()
            return r.content
        except Missing:
            raise
        except Exception as e:
            if k == tries - 1:
                raise
            print(f'{dt.datetime.now():%m-%d %H:%M} [重试 {k + 1}] {url.rsplit("/", 1)[-1]}：{str(e)[:100]}', flush=True)
            time.sleep(min(15 * 2 ** k, 600))


def ranges_of(idx_text, names):
    """.idx 每行 "序号:起始字节:d=…:变量:层次:时效:"，返回 {变量: (起, 止)}；只取 surface 层（涌浪分量不用）。"""
    rows = [ln.split(':') for ln in idx_text.strip().splitlines()]
    out = {}
    for i, r in enumerate(rows):
        if r[3] in names and r[4] == 'surface' and r[3] not in out:
            end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else ''
            out[r[3]] = (int(r[1]), end)
    missing = set(names) - set(out)
    if missing:
        raise KeyError(f'索引里没有 {missing}')
    return out


def decode_at(msg, pts):
    """解码一条 GRIB2 记录，返回各点的值与插值方式（0 双线性、1 部分角点、2 最近海上格点、3 无值）。"""
    import eccodes
    with _GRIB_LOCK:
        gid = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(gid, 'Ni'), eccodes.codes_get(gid, 'Nj')
        lat0 = eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
        lon0 = eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
        dlat = eccodes.codes_get(gid, 'jDirectionIncrementInDegrees')
        dlon = eccodes.codes_get(gid, 'iDirectionIncrementInDegrees')
        jpos = eccodes.codes_get(gid, 'jScansPositively')
        vals = eccodes.codes_get_values(gid).astype(float)
        if eccodes.codes_get(gid, 'bitmapPresent'):
            vals[vals == eccodes.codes_get(gid, 'missingValue')] = np.nan
    finally:
        eccodes.codes_release(gid)
    f = vals.reshape(nj, ni)
    out, how = [], []
    for lat, lon in pts:
        y = (lat - lat0) / dlat if jpos else (lat0 - lat) / dlat
        x = ((lon - lon0) % 360) / dlon
        j0, i0 = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - j0, x - i0
        cj = [min(max(j0, 0), nj - 1), min(max(j0 + 1, 0), nj - 1)]
        ci = [i0 % ni, (i0 + 1) % ni]
        v = np.array([f[cj[0], ci[0]], f[cj[0], ci[1]], f[cj[1], ci[0]], f[cj[1], ci[1]]])
        w = np.array([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx])
        ok = np.isfinite(v)
        if ok.all():
            out.append(float(v @ w)); how.append(0)
        elif ok.any() and w[ok].sum() > 0:
            out.append(float(v[ok] @ w[ok] / w[ok].sum())); how.append(1)
        else:
            best, bd = np.nan, np.inf
            for dj in range(-2, 4):
                for di in range(-2, 4):
                    jj, ii = j0 + dj, (i0 + di) % ni
                    if 0 <= jj < nj and np.isfinite(f[jj, ii]):
                        d = (jj - y) ** 2 + (((ii - x + ni / 2) % ni) - ni / 2) ** 2
                        if d < bd:
                            best, bd = float(f[jj, ii]), d
            out.append(best); how.append(2 if np.isfinite(best) else 3)
    return np.array(out), how


def read_whole(ses, u, names, pts):
    """没有 .idx 时（10-27 发现 2021–2023 年的 mean 文件缺索引，成员文件不缺）整份读入，按 GRIB 参数编号找记录。"""
    import eccodes
    blob, pos, res, hows = fetch(ses, u), 0, {}, {}
    while pos < len(blob) and len(res) < len(names):
        n = int.from_bytes(blob[pos + 8:pos + 16], 'big')
        msg = blob[pos:pos + n]
        with _GRIB_LOCK:
            gid = eccodes.codes_new_from_message(msg)
        key = tuple(eccodes.codes_get(gid, k) for k in ('discipline', 'parameterCategory', 'parameterNumber'))
        eccodes.codes_release(gid)
        v = next((v for v in names if PARAM[v] == key and v not in res), None)
        if v:
            res[v], hows[v] = decode_at(msg, pts)
        pos += n
    if set(names) - set(res):
        raise KeyError(f'{u} 里没有 {set(names) - set(res)}')
    return res, len(blob), hows


def read_record(ses, day, member, fh, names, pts, bases=(BUCKET,)):
    """返回 ({变量: 各点值}, 流量字节, {变量: 插值方式})。bases 依次尝试（镜像在前、AWS 在后）。"""
    u = None
    for base in bases:
        try:
            rg = ranges_of(fetch(ses, url_of(day, member, fh, base) + '.idx').decode(), names)
            u = url_of(day, member, fh, base)
            break
        except Missing:
            pass
    if u is None:                                         # 各处都没有索引：整份读
        for base in bases:
            try:
                return read_whole(ses, url_of(day, member, fh, base), names, pts)
            except Missing:
                pass
        raise Missing(url_of(day, member, fh))
    groups = []                                           # 索引里相邻的记录合并成一次请求
    for v, r in sorted(rg.items(), key=lambda x: x[1][0]):
        if groups and groups[-1][1] != '' and groups[-1][1] + 1 == r[0]:
            groups[-1] = (groups[-1][0], r[1], groups[-1][2] + [(v, r)])
        else:
            groups.append((r[0], r[1], [(v, r)]))
    res, nbytes, hows = {}, 0, {}
    for a, b, items in groups:
        blob = fetch(ses, u, (a, b))
        nbytes += len(blob)
        for v, r in items:
            s = r[0] - a
            if blob[s:s + 4] != b'GRIB':                  # 索引的起始偏移都不对：整份读
                return read_whole(ses, u, names, pts)
            n = int.from_bytes(blob[s + 8:s + 16], 'big')  # 记录长度以 GRIB2 第 0 段为准（10-28：20221019 的 .idx 偏移差几个字节，按索引切会截掉结尾 7777）
            msg = blob[s:s + n]
            if len(msg) < n:
                msg = fetch(ses, u, (r[0], r[0] + n - 1))
                nbytes += len(msg)
            res[v], hows[v] = decode_at(msg, pts)
    return res, nbytes, hows


def to_3h(vals, directional=False):
    """各时效（FHOURS）的值 → 逐 3 小时（3–384 h）；方向按 sin / cos 插值。"""
    src, dst = np.array(FHOURS, float), np.arange(3, 385, 3, dtype=float)
    if not directional:
        return np.interp(dst, src, vals, left=np.nan, right=np.nan)
    r = np.deg2rad(vals)
    s, c = np.interp(dst, src, np.sin(r)), np.interp(dst, src, np.cos(r))
    return (np.rad2deg(np.arctan2(s, c)) + 360) % 360


def day_job(day, members, stations, member_vars, mean_vars, fhours, threads, bases=(BUCKET,)):
    """一个起报日：并行读全部（成员, 时效）记录，返回 (DataFrame | None, 流量, 插值方式, 说明)。"""
    pts = [(s['lat'], s['lon']) for s in stations]
    ses = requests.Session()
    ses.headers['User-Agent'] = 'research'
    ses.mount('https://', requests.adapters.HTTPAdapter(pool_connections=threads, pool_maxsize=threads))
    tasks = [(m, fh, member_vars) for m in members for fh in fhours]
    if mean_vars:
        tasks += [('mean', fh, mean_vars) for fh in fhours]
    got, nbytes, hows = {}, 0, {}
    for base in bases:
        try:
            fetch(ses, url_of(day, members[0], fhours[0], base) + '.idx')
            break
        except Missing:
            pass
    else:
        return None, 0, {}, '存档缺失'
    with cf.ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(read_record, ses, day, m, fh, names, pts, bases): (m, fh, names) for m, fh, names in tasks}
        bad = []
        for fu in cf.as_completed(futs):
            m, fh, names = futs[fu]
            try:
                res, b, h = fu.result()
            except Missing:
                continue
            except Exception as e:                        # 个别记录损坏（10-28：20221019 报 ecCodes "Key/value not found"）：倒序换镜像再读一次，仍不行就当这个时效缺失
                try:
                    res, b, h = read_record(ses, day, m, fh, names, pts, bases[::-1])
                except Exception:
                    bad.append(f'{m}/f{fh:03d}（{str(e)[:40]}）')
                    continue
            got[(m, fh)] = res
            nbytes += b
            for v, hh in h.items():
                hows.setdefault(v, hh)
    ses.close()
    n_need = len(fhours)
    fh_full = fhours == FHOURS
    rows, note = [], ([f'损坏记录 {len(bad)} 条：' + '，'.join(bad[:5])] if bad else [])
    for m in members:
        have = sum((m, fh) in got for fh in fhours)
        if have < n_need:
            note.append(f'{m} 缺 {n_need - have} 个时效')
            if have < 0.9 * n_need:
                continue
        for k, st in enumerate(stations):
            series = {}
            for v in member_vars + mean_vars:
                src = m if v in member_vars else 'mean'
                x = np.array([got[(src, fh)][v][k] if (src, fh) in got else np.nan for fh in fhours])
                series[COL[v]] = to_3h(x, v in DIRECTIONAL) if fh_full else x
            lead = np.arange(3, 385, 3) if fh_full else np.array(fhours)
            d = pd.DataFrame(series)
            for c in ('hs', 'tr', 'th1m', 'tm02'):
                if c not in d:
                    d[c] = np.nan
            d.insert(0, 'member', m)
            d.insert(0, 'issue_time', pd.Timestamp(day).strftime('%Y-%m-%d'))
            d.insert(0, 'station', st['station'])
            d.insert(0, 'valid_time', (pd.Timestamp(day) + pd.to_timedelta(lead, unit='h')).strftime('%Y-%m-%d %H:%M:%S'))
            d['lat'], d['lon'] = st['lat'], st['lon']
            d['native'] = np.isin(lead, fhours).astype(int)
            rows.append(d[['valid_time', 'station', 'issue_time', 'member', 'hs', 'tr', 'th1m', 'tm02', 'lat', 'lon', 'native']])
    if not rows:
        return None, nbytes, hows, '全部成员缺失；' + '；'.join(note)
    return pd.concat(rows, ignore_index=True), nbytes, hows, '；'.join(note)


def station_list(ids):
    if ids == ['all']:
        ids = [k for k, s in SITES.items() if s.get('buoy')]
    out = []
    for i in ids:
        s = SITES[i]
        b = s.get('buoy') or {}
        lat, lon = b.get('lat', s['point'][0]), b.get('lon', s['point'][1])
        out.append(dict(site=i, station=str(b.get('station', i)), lat=float(lat), lon=float(lon) % 360))
    return out


def save(out_dir, df, day):
    for st, g in df.groupby('station'):
        d = os.path.join(out_dir, st)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f'{day[:4]}.csv.gz')
        g.round(4).to_csv(p, mode='a', header=not os.path.exists(p), index=False, compression='gzip')


def check_station(n, stations, threads):
    """格点插值与业务站点序列（wave/station/ 的 31 成员平均 Hs）比较，只看 G、H（站点序列不含韩国站）。
    这一项只衡量"格点插值 vs 站点输出"的差别，回算用的是站点输出。"""
    stations = [s for s in stations if s['station'] in ('52200', '51207')]
    days = pd.date_range('2021-01-06', '2025-12-31', freq='7D')
    days = [d.strftime('%Y%m%d') for d in days[np.linspace(0, len(days) - 1, n).round().astype(int)]]
    ses = requests.Session()
    ses.headers['User-Agent'] = 'research'
    pts = [(s['lat'], s['lon']) for s in stations]
    recs = []
    for day in days:
        try:
            tar = tarfile.open(fileobj=io.BytesIO(fetch(ses, f'{BUCKET}gefs.{day}/00/wave/station/gefs.wave.t00z.station_tar')))
        except Missing:
            print(f'{day} 没有站点序列，跳过', flush=True)
            continue
        ts = {}
        for s in stations:
            txt = tar.extractfile(f'gefs.wave.{s["station"]}.ts').read().decode(errors='ignore').splitlines()
            for ln in txt:
                p = ln.split()
                if len(p) >= 3 and p[0].isdigit() and len(p[0]) == 8:
                    ts[(s['station'], pd.Timestamp(p[0]) + pd.Timedelta(hours=int(p[1])))] = float(p[2])
        with cf.ThreadPoolExecutor(threads) as ex:
            futs = {ex.submit(read_record, ses, day, 'mean', fh, ['HTSGW'], pts): fh for fh in FHOURS}
            for fu in cf.as_completed(futs):
                fh = futs[fu]
                res = fu.result()[0]['HTSGW']
                t = pd.Timestamp(day) + pd.Timedelta(hours=fh)
                for k, s in enumerate(stations):
                    if (s['station'], t) in ts:
                        recs.append(dict(day=day, station=s['station'], lead_h=fh, grid=res[k], station_ts=ts[(s['station'], t)]))
        print(f'{day} 完成', flush=True)
    d = pd.DataFrame(recs)
    out = {}
    for st, g in d.groupby('station'):
        out[st] = dict(n=len(g), mean_grid=g.grid.mean(), mean_station=g.station_ts.mean(), ratio=g.grid.mean() / g.station_ts.mean(),
                       bias_m=(g.grid - g.station_ts).mean(), rmse_m=float(np.sqrt(((g.grid - g.station_ts) ** 2).mean())),
                       corr=g[['grid', 'station_ts']].corr().iloc[0, 1])
    path = os.path.join(ROOT, 'results', 'gefs_operational_grid_vs_station.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(days=days, by_station=out), open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(f'→ {path}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--sites', nargs='+', default=['G', 'K', 'H'])
    p.add_argument('--start', default=FIRST_DAY)
    p.add_argument('--end', default='2024-12-31', help='默认到 2024 年底：本机 ERA5 站点序列（过去 7 天输入）止于 2024-12-31')
    p.add_argument('--daily', action='store_true', help='所有日期（默认只取周三）')
    p.add_argument('--members', type=int, default=11, help='c00 + p01… 共几个成员（最多 31）')
    p.add_argument('--member-vars', nargs='+', default=['HTSGW', 'MWSPER', 'WWSDIR'])
    p.add_argument('--mean-vars', nargs='*', default=[], help='从集合平均文件取的变量（mean 文件只有 HTSGW、PERPW、DIRPW 等）')
    p.add_argument('--procs', type=int, default=4, help='同时处理的起报日数（进程）')
    p.add_argument('--threads', type=int, default=12, help='每个起报日的并行请求数')
    p.add_argument('--out', default='processed/gefs_operational')
    p.add_argument('--mirror', choices=list(MIRRORS), default='aws', help='gcs：先取 Google Cloud 镜像，没有的退回 AWS')
    p.add_argument('--test', action='store_true')
    p.add_argument('--check-station', type=int, default=0)
    a = p.parse_args()
    stations = station_list(a.sites)
    if a.check_station:
        return check_station(a.check_station, stations, a.threads)
    members = (['c00'] + [f'p{i:02d}' for i in range(1, 31)])[:a.members]
    mean_vars = [v for v in a.mean_vars if v not in a.member_vars]
    fhours = FHOURS
    if a.test:
        a.start = a.end = FIRST_DAY
        members, fhours, a.out = members[:2], FHOURS[:8], a.out + '_test'
    out_dir = os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    done_file, miss_file = os.path.join(out_dir, '_done.txt'), os.path.join(out_dir, '_missing.txt')
    done = set(open(done_file, encoding='utf-8').read().split()) if os.path.exists(done_file) else set()
    missing = set(open(miss_file, encoding='utf-8').read().split()) if os.path.exists(miss_file) else set()
    days = pd.date_range(max(a.start, FIRST_DAY), a.end, freq='1D')
    if not a.daily:
        days = days[days.dayofweek == 2]
    todo = [d.strftime('%Y%m%d') for d in days if d.strftime('%Y%m%d') not in done | missing]
    print(f'{dt.datetime.now():%m-%d %H:%M} 站点 {[s["station"] for s in stations]}，成员 {len(members)} 个，时效 {len(fhours)} 个；'
          f'待处理起报日 {len(todo)} 个（已完成 {len(done)}、存档缺失 {len(missing)}）→ {out_dir}', flush=True)
    t0, total, points = time.time(), 0, {}
    pts_file = os.path.join(out_dir, '_points.json')
    with cf.ProcessPoolExecutor(a.procs) as ex:
        bases = (MIRRORS[a.mirror],) + ((BUCKET,) if a.mirror != 'aws' else ())
        futs = {ex.submit(day_job, day, members, stations, a.member_vars, mean_vars, fhours, a.threads, bases): day for day in todo}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            day = futs[fu]
            try:
                df, nbytes, hows, note = fu.result()
            except Exception as e:
                print(f'{dt.datetime.now():%m-%d %H:%M} [失败，下次重跑会再试] {day}：{str(e)[:150]}', flush=True)
                continue
            total += nbytes
            if df is None:
                if note == '存档缺失':
                    open(miss_file, 'a', encoding='utf-8').write(day + '\n')
                print(f'{dt.datetime.now():%m-%d %H:%M} {day}：{note}', flush=True)
                continue
            save(out_dir, df, day)
            open(done_file, 'a', encoding='utf-8').write(day + '\n')
            if not points and hows:
                how_name = {0: '双线性', 1: '部分角点（近岸）', 2: '最近海上格点', 3: '无值'}
                points = {s['station']: {v: how_name[h[k]] for v, h in hows.items()} for k, s in enumerate(stations)}
                json.dump(dict(stations=stations, interpolation=points), open(pts_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                print(f'插值方式：{points}', flush=True)
            el = time.time() - t0
            print(f'{dt.datetime.now():%m-%d %H:%M} [{i}/{len(todo)}] {day} 完成 {len(df)} 行{("（" + note + "）") if note else ""}；'
                  f'累计 {total / 1e9:.1f} GB，{el / 60:.0f} 分钟，预计还要 {el / i * (len(todo) - i) / 3600:.1f} 小时', flush=True)
    print(f'{dt.datetime.now():%m-%d %H:%M} 结束。整理：python prepare_reforecast_dataset.py G K H --sources {a.out.split("/")[-1]} --suffix _oper', flush=True)


if __name__ == '__main__':
    main()
