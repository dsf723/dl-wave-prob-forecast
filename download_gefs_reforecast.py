"""NOAA GEFSv12 海浪集合回算（2000–2019，WAVEWATCH III）在浮标站上的历史预报提取。

资料：AWS 公开桶 noaa-nws-gefswaves-reforecast-pds，GEFSv12/reforecast/<年>/<日期>/station/gefs.wave.<日期>.<成员>.tab.nc
  - 每天 1 次起报；平日 5 个成员（c00、p01–p04）、16 天；**周三 11 个成员（c00、p01–p10）、35 天**；
  - tab.nc 含 658 个站点逐时预报：hs 有效波高、tr 平均周期（WW3 定义待核实）、fp 谱峰频率、th1m/th1p 平均/谱峰方向、
    sth1m/sth1p 方向分布宽度、lm 平均波长。
做法（默认，10-16 起）：用 HTTP 分段读取（Range 请求）只读 station_name、time、经纬度首行和 VARS_RANGE
  （hs、tr、fp、th1m），不下载整个文件。文件按"每个时刻一块、含全部 658 个站点"压缩存储，取少数站点也要读完
  整个变量，每个变量约 2 MB：实测约 8.4 MB/文件，而整文件约 18.9 MB（10-16 前整文件下载了 5,570 个，约 105 GB）。
  变量含义（10-16 用关岛 2004–2008 核实）：tr 与 ERA5 能量周期 Te 比值 1.02、相关 0.87，即 **tr = 能量周期 Te**
  （可直接查功率矩阵 Hs × Te）；1/fp = 谱峰周期 Tp，与浮标 DPD 比值 1.01；th1m = 平均波向。
  输出列与整文件模式相同，th1p、sth1m、lm 为空（功率估算用不到）。
做法（--full）：整文件下载，提取全部 7 个变量（流量约 8.5 倍）。
可随时中断，已完成的（日期, 成员）记在 processed/gefs_reforecast/_done.txt，重跑会跳过。
用法：
  python download_gefs_reforecast.py            # 默认只下周三（11 成员、35 天，覆盖 1–30 天中期），只取 hs
  python download_gefs_reforecast.py --full     # 整文件、全部变量
  python download_gefs_reforecast.py --daily    # 所有日期（平日 5 成员、16 天），数据量约再多一倍
  python download_gefs_reforecast.py --daily --weekdays-only --years 2000 2019 --vars hs tr th1m
        # 10-21 新增：只下平日（周三已有）、指定年份和变量；平日文件 3 个变量约 3.4 MB/个，2000–2019 约 31,000 个、约 106 GB
  中断后用同一条命令重跑即可续传（按 <out>/_done.txt 跳过已完成的起报日 × 成员）。被中断的那一批里，个别文件可能已写入
  csv 却没记进 _done.txt，重跑会重复写入几行，整理数据时按（站点, 起报, 成员, 时刻）去重即可。
  --procs N、--batch-days N（10-22 加）：分段读取的进程数和一次并行的起报日数（默认 12、4），服务器上可用 24、8 提速。
"""
import concurrent.futures as cf
import datetime as dt
import os
import re
import sys
import tempfile
import threading
import time
import io
import urllib.request

import h5py
import numpy as np
import requests
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'processed', 'gefs_reforecast')
BUCKET = 'https://noaa-nws-gefswaves-reforecast-pds.s3.amazonaws.com/'
# 浮标验证站与附近站点：G 关岛 52200（52202 备选）；H 夏威夷 WETS 51207（51210、51202 备选）；
# K 韩国马罗岛 22107（22184、22187 备选）；C 台湾海峡附近的台湾浮标 46757B、46694A、C6W08
STATIONS = ['52200', '52202', '51207', '51210', '51202', '22107', '22184', '22187', '46757B', '46694A', 'C6W08']
# 10-22 扩充：回算站点表里关岛 / 夏威夷 / 韩国 / 台湾附近的全部浮标点（每个文件读取的流量与站点数无关，一次取全）。
# 扩充后请用新的 --out 目录重跑（旧目录的 _done.txt 会把已做过的日期跳过）。
STATIONS += ['52211',                                                                  # 塞班 Tanapag
             '51201', '51203', '51204', '51205', '51206', '51208', '51211', '51212', '51213', '51003', '51200',  # 夏威夷
             '22101', '22102', '22103', '22104', '22105', '22106', '22108', '22183', '22185', '22186',
             '22188', '22189', '22190', '21229',                                       # 韩国 KMA（21229 为郁陵岛附近）
             '46699A', '46706A', '46708A', '46714D', '46735A', '46759A', '46778A', '46787A', 'C6V27', 'WRA007']  # 台湾
VARS = ['hs', 'tr', 'fp', 'th1m', 'th1p', 'sth1m', 'lm']
VARS_RANGE = ['hs', 'tr', 'fp', 'th1m']           # 分段读取模式取的变量：波高、能量周期、谱峰频率、平均波向
WORKERS = 4                             # 整文件模式的线程数
PROCS = 12                              # 分段读取模式的进程数：h5py 有全局锁，多线程读不能并行，必须用多进程（10-16 实测）
BATCH_DAYS = 4                          # 一次并行处理的起报日数
BLOCK = 128 * 1024                      # 分段读取的块大小（实测 128 KB：约 2.9 MB/文件、10 s/文件；512 KB 时 4.7 MB）
_lock = threading.Lock()
_bytes = [0]                            # 本次运行累计流量（字节）


class RangeFile(io.RawIOBase):
    """只按需读取远程文件的若干字节段（HTTP Range），供 h5py 打开。"""

    def __init__(self, url):
        self.url, self.pos, self.cache, self.ses = url, 0, {}, requests.Session()
        self.ses.headers['User-Agent'] = 'research'
        self.size = int(self.ses.head(url, timeout=60).headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            a = i * BLOCK
            r = self.ses.get(self.url, headers={'Range': f'bytes={a}-{min(a + BLOCK, self.size) - 1}'}, timeout=120)
            r.raise_for_status()
            self.cache[i] = r.content
            with _lock:
                _bytes[0] += len(r.content)
        return self.cache[i]

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out, p = bytearray(), self.pos
        while len(out) < n:
            i = p // BLOCK
            d = self._block(i)
            take = d[p - i * BLOCK: p - i * BLOCK + n - len(out)]
            out += take
            p += len(take)
        buf[:n] = out
        self.pos += n
        return n

    def close(self):
        self.ses.close()
        super().close()


def _decode_time(var):
    unit, ref = var.attrs['units'].decode() if isinstance(var.attrs['units'], bytes) else var.attrs['units'], None
    step, ref = unit.split(' since ')
    scale = {'seconds': 's', 'second': 's', 'minutes': 'min', 'hours': 'h', 'days': 'D'}[step.strip()]
    return pd.Timestamp(ref.strip().replace('Z', '')) + pd.to_timedelta(var[:], unit=scale)


def extract_range(day, member):
    """分段读取：只取 VARS_RANGE。"""
    url = f'{BUCKET}GEFSv12/reforecast/{day[:4]}/{day}/station/gefs.wave.{day}.{member}.tab.nc'
    for k in range(6):
        try:
            raw = RangeFile(url)
            try:
                with h5py.File(io.BufferedReader(raw, buffer_size=BLOCK), 'r') as h:
                    names = [b''.join(r).decode().strip() for r in h['station_name'][:]]
                    idx = [names.index(s) for s in STATIONS if s in names]
                    t = _decode_time(h['time'])
                    vals = {v: h[v][:, :][:, idx] for v in VARS_RANGE}
                    lat = h['latitude'][0, :][idx]
                    lon = h['longitude'][0, :][idx]
            finally:
                raw.close()
            break
        except Exception:
            if k == 5:
                raise
            time.sleep(min(600, 30 * 2 ** k))
    rows = []
    for j, i in enumerate(idx):
        d = pd.DataFrame({v: vals[v][:, j] if v in vals else np.nan for v in VARS}, index=pd.DatetimeIndex(t, name='valid_time'))
        for v in VARS_RANGE:
            d[v] = d[v].where(np.abs(d[v]) < 1e3)          # 缺测填充值
        d.insert(0, 'member', member)
        d.insert(0, 'issue_time', pd.Timestamp(day))
        d.insert(0, 'station', names[i])
        d['lat'], d['lon'] = float(lat[j]), float(lon[j])
        rows.append(d.reset_index())
    return pd.concat(rows) if rows else None


def http(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research'}), timeout=timeout).read()


def members_of(day):
    """列出某天的成员。网络中断（10-16 凌晨出现过约 9 小时断网）时每 10 分钟重试，最多约 12 小时。"""
    for k in range(72):
        try:
            x = http(f'{BUCKET}?list-type=2&prefix=GEFSv12/reforecast/{day[:4]}/{day}/station/', 60).decode()
            return sorted(set(re.findall(rf'gefs\.wave\.{day}\.(\w+)\.tab\.nc', x)))
        except Exception as e:
            print(f'{dt.datetime.now():%m-%d %H:%M} [网络重试 {k + 1}] 列出 {day} 成员失败：{str(e)[:120]}', flush=True)
            time.sleep(600)
    raise RuntimeError(f'列出 {day} 成员连续失败约 12 小时')


def extract(day, member):
    url = f'{BUCKET}GEFSv12/reforecast/{day[:4]}/{day}/station/gefs.wave.{day}.{member}.tab.nc'
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'tab.nc')
        for k in range(4):
            try:
                data = http(url)
                open(p, 'wb').write(data)
                with _lock:
                    _bytes[0] += len(data)
                break
            except Exception:
                if k == 3:
                    raise
                time.sleep(20 * (k + 1))
        with xr.open_dataset(p) as ds:
            names = [b''.join(r).decode().strip() for r in ds['station_name'].values]
            idx = [names.index(s) for s in STATIONS if s in names]
            sub = ds[VARS + ['latitude', 'longitude']].isel(station=idx).load()
    rows = []
    for j, i in enumerate(idx):
        d = pd.DataFrame({v: sub[v].isel(station=j).values for v in VARS}, index=pd.DatetimeIndex(sub['time'].values, name='valid_time'))
        d.insert(0, 'member', member)
        d.insert(0, 'issue_time', pd.Timestamp(day))
        d.insert(0, 'station', names[i])
        d['lat'] = float(sub['latitude'].isel(station=j, time=0))
        d['lon'] = float(sub['longitude'].isel(station=j, time=0))
        rows.append(d.reset_index())
    return pd.concat(rows) if rows else None


def save(df, day):
    with _lock:
        for st, g in df.groupby('station'):
            d = os.path.join(OUT, st)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f'{day[:4]}.csv.gz')
            g.to_csv(p, mode='a', header=not os.path.exists(p), index=False, compression='gzip')


def _range_job(day, member, variables=None):
    """子进程里执行分段读取，返回 (结果, 本文件流量字节)。子进程不继承主进程改过的变量列表，所以显式传入。"""
    if variables:
        VARS_RANGE[:] = variables
    b0 = _bytes[0]
    df = extract_range(day, member)
    return df, _bytes[0] - b0


def main():
    """10-16 提速：分段读取每个文件要几十次小请求，瓶颈是网络往返延迟；h5py 全局锁使多线程无效，所以
    （1）分段读取改用 PROCS 个进程、一次处理 BATCH_DAYS 个起报日；（2）周三 11 个成员都已完成的日子不再列目录。"""
    global OUT, PROCS, BATCH_DAYS
    os.makedirs(OUT, exist_ok=True)
    if '--procs' in sys.argv:                # 10-22：服务器上可加大并行，如 --procs 24 --batch-days 8
        PROCS = int(sys.argv[sys.argv.index('--procs') + 1])
    if '--batch-days' in sys.argv:
        BATCH_DAYS = int(sys.argv[sys.argv.index('--batch-days') + 1])
    daily = '--daily' in sys.argv
    func = extract if '--full' in sys.argv else extract_range
    args = sys.argv[1:]
    if '--vars' in args:                     # e.g. --vars hs tr th1m
        i = args.index('--vars'); chosen = []
        for v in args[i + 1:]:
            if v.startswith('--'):
                break
            chosen.append(v)
        VARS_RANGE[:] = chosen
    if '--out' in args:                      # e.g. --out processed/gefs_reforecast_weekday（相对项目根目录）
        OUT = os.path.join(ROOT, args[args.index('--out') + 1])
        os.makedirs(OUT, exist_ok=True)
        done_file = os.path.join(OUT, '_done.txt')
        done = set(open(done_file, encoding='utf-8').read().split()) if os.path.exists(done_file) else set()
    year0, year1 = 2000, 2019
    if '--years' in args:
        i = args.index('--years'); year0, year1 = int(args[i + 1]), int(args[i + 2])
    done_file = os.path.join(OUT, '_done.txt')
    done = set(open(done_file, encoding='utf-8').read().split()) if os.path.exists(done_file) else set()
    per_day = {}
    for k in done:
        per_day[k[:8]] = per_day.get(k[:8], 0) + 1
    days = pd.date_range(f'{year0}-01-01', f'{year1}-12-31', freq='1D')
    if not daily:
        days = days[days.dayofweek == 2]                         # 周三
    elif '--weekdays-only' in args:
        days = days[days.dayofweek != 2]                         # 平日（周三已单独下载）
    todo = [d.strftime('%Y%m%d') for d in days if per_day.get(d.strftime('%Y%m%d'), 0) < (11 if d.dayofweek == 2 else 5)]
    print(f'{dt.datetime.now():%m-%d %H:%M} 待处理起报日 {len(todo)} 个（已完成 {len(days) - len(todo)} 个）', flush=True)
    t0, n = time.time(), 0
    for b in range(0, len(todo), BATCH_DAYS):
        batch = todo[b:b + BATCH_DAYS]
        jobs = [(day, m) for day in batch for m in members_of(day) if f'{day}_{m}' not in done]
        pool = cf.ThreadPoolExecutor(WORKERS) if func is extract else cf.ProcessPoolExecutor(PROCS)
        with pool as ex:
            futs = {(ex.submit(extract, day, m) if func is extract else ex.submit(_range_job, day, m, list(VARS_RANGE))): (day, m)
                    for day, m in jobs}
            for f in cf.as_completed(futs):
                day, m = futs[f]
                try:
                    df = f.result()
                    if func is not extract:
                        df, nb = df
                        _bytes[0] += nb
                    if df is not None:
                        save(df, day)
                    with _lock, open(done_file, 'a', encoding='utf-8') as fh:
                        fh.write(f'{day}_{m}\n')
                    n += 1
                except Exception as e:
                    print(f'[失败] {day} {m}：{str(e)[:150]}（重跑会补）', flush=True)
        print(f'{dt.datetime.now():%m-%d %H:%M} 起报 {batch[0]}–{batch[-1]}：{len(jobs)} 个文件完成；累计 {n} 个文件，'
              f'平均 {(time.time() - t0) / max(n, 1):.1f} s/文件，累计流量 {_bytes[0] / 1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    if sys.stdout is None:                    # pythonw（无控制台）运行时：输出写入日志文件
        os.makedirs(OUT, exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(OUT, '_run_log.txt'), 'a', encoding='utf-8', buffering=1)
    else:
        sys.stdout.reconfigure(encoding='utf-8')
    main()
