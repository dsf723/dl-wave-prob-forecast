"""检查服务器上提取的 GEFS 回算文件（每站每年一个 .csv.gz，由多次追加写成）有没有损坏。

下载进程被强制结束时，正在追加的那一段可能只写了一半；之后续传又在后面追加，普通读取会在坏段处报错。
本脚本逐个文件试读；读不了的用 prepare_reforecast_dataset.read_gz_tolerant 跳过坏段后读回，报告能读回多少行。
坏段里的行属于当时没记进 _done.txt 的（日期, 成员），下载续传 / 补漏时会重新写入，整理数据时还会去重。
用法：python check_reforecast_files.py [gefs_reforecast_weekday gefs_reforecast_weekday_early gefs_reforecast_v2]
"""
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from prepare_reforecast_dataset import read_gz_tolerant, COLS   # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    dirs = sys.argv[1:] or ['gefs_reforecast_weekday', 'gefs_reforecast_weekday_early', 'gefs_reforecast_v2']
    for src in dirs:
        files = sorted(glob.glob(os.path.join(ROOT, 'processed', src, '*', '*.csv.gz')))
        bad = []
        for f in files:
            try:
                pd.read_csv(f, usecols=COLS[:1])
            except Exception as e:
                d, n = read_gz_tolerant(f)
                bad.append((os.path.relpath(f, ROOT), n, len(d), str(e)[:50]))
        print(f'{src}：{len(files)} 个文件，损坏 {len(bad)} 个')
        for f, n, rows, err in bad:
            print(f'    {f}：跳过 {n} 段坏数据，读回 {rows} 行（{err}）')


if __name__ == '__main__':
    main()
