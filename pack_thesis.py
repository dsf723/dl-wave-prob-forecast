"""收尾打包：把服务器上写论文和存档有用的东西打成压缩包，下载回本机。只读取，不删除、不改动任何已有文件。

与 pack_release.py 的区别：复现包只含 G、K、H 三个冻结运行，用来核对可复现性；这里范围更大（台湾海峡、全部对比试验的摘要、全部结果）。
只打包白名单里的文件；文件名含 token、credential、secret、cdsapirc、netrc 的一律跳过；不含 PDF、Word、参考文献。

三个包（都在 --out 目录，默认 thesis_pack/）：
  core   必需。代码、配置、metadata、results/；5 个最终运行的全部文件（GKH Transformer、GRU、周期多任务，TWS Transformer、GRU）；
         其余运行的配置、权重、摘要成绩和日志，不含逐时刻明细；G、K、H、台湾海峡 5 站和 A、B、C 的整理数据，浮标、台风标签；环境记录。
  gefs   建议。服务器提取的站点级序列：gefs_reforecast_v2（周三 47 站）、gefs_operational。重新提取要几天，所以单独存档。已是 gz，不再压缩。
  runs   可选。其余运行的逐时刻明细（case_scores.csv、*_val.npz），只有以后要重算这些试验的自助区间才用得到。
不打包：原始 GRIB（可重新下载）、平日回算 gefs_reforecast_weekday*（10-24 dailymix 无显著收益，未采用）。

用法：python pack_thesis.py --dry-run             只列出各包大小、磁盘剩余和缺失项，不写文件
      python pack_thesis.py core [gefs] [runs]    打包，最后打印每个包的 SHA256
      python pack_thesis.py --verify 包路径         在任何机器上核对包内每个文件的 SHA256（不解压到磁盘）
"""
import argparse
import datetime as dt
import glob
import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = 'wave_thesis'
FINAL = ['GKH_h15_transformer_pretrain', 'GKH_h15_gru_pretrain', 'GKH_h15_te_transformer_pretrain', 'TWS_h15_transformer_pretrain', 'TWS_h15_gru_pretrain']
TW = ['TWC6W08', 'TW46757B', 'TW46735A', 'TW46787A', 'TW46778A']
DETAIL = re.compile(r'(^|/)case_scores\.csv$|_val\.npz$')
SKIP = re.compile(r'token|credential|secret|cdsapirc|netrc|__pycache__|_resolved', re.I)
CODE = ['*.py', '*.sh', '*.md', 'requirements*.txt', 'metadata/**/*', 'model_configs/*.json', 'outputs/wave_forecast/wavecast/*.py', 'outputs/wave_forecast/tests/*.py',
        'outputs/wave_forecast/*.md', 'outputs/wave_forecast/requirements.txt', 'outputs/wave_forecast/config.json']
DATA = [f'processed/site_{s}{x}/**/*' for s in 'GKH' for x in ('', '_reforecast', '_era5pre', '_reforecast_oper', '_reforecast_oper_hsbridge')]
DATA += [f'processed/site_{s}{x}/**/*' for s in TW for x in ('', '_reforecast', '_era5pre')]
DATA += [f'processed/site_{s}/**/*' for s in 'ABC']
DATA += [f'processed/buoy/{s}_*.csv' for s in ['G', 'K', 'H'] + TW] + [f'processed/tracks/tc_hourly_{s}.csv' for s in ['G', 'K', 'H'] + TW] + ['processed/buoy_spectral/**/*']
GEFS = ['processed/gefs_reforecast_v2/**/*', 'processed/gefs_operational/**/*']
EXPECT = [f'model_runs/{r}/{x}' for r in FINAL for x in ('selection.json', 'test/quantiles.npz', 'test/case_scores.csv')]
EXPECT += ['results/taiwan_strait/validation_report.md', 'results/taiwan_strait/test_report.md']
EXPECT += [f'processed/site_{s}_reforecast_oper/manifest.json' for s in 'GKH']


def files(patterns):
    out = set()
    for p in patterns:
        for f in glob.glob(os.path.join(ROOT, p), recursive=True):
            rel = os.path.relpath(f, ROOT).replace('\\', '/')
            if os.path.isfile(f) and not SKIP.search(rel):
                out.add(rel)
    return out


def groups():
    all_runs = files(['model_runs/**/*'])
    final = {f for f in all_runs if f.split('/')[1] in FINAL}
    detail = {f for f in all_runs - final if DETAIL.search(f)}
    return {'core': sorted(files(CODE) | files(['results/**/*']) | files(DATA) | (all_runs - detail)), 'gefs': sorted(files(GEFS)), 'runs': sorted(detail)}


def environment():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
        except Exception as e:
            return f'（无法运行 {cmd[0]}：{e}）\n'
    return {'env/environment.txt': f'记录时间 {dt.datetime.now():%Y-%m-%d %H:%M}\nPython {sys.version}\n系统 {platform.platform()}\n',
            'env/pip_freeze.txt': run([sys.executable, '-m', 'pip', 'freeze']), 'env/nvidia_smi.txt': run(['nvidia-smi'])}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def add_text(tar, name, text):
    data = text.encode('utf-8')
    info = tarfile.TarInfo(f'{TOP}/{name}')
    info.size, info.mtime = len(data), int(dt.datetime.now().timestamp())
    tar.addfile(info, io.BytesIO(data))
    return f'{hashlib.sha256(data).hexdigest()}  {name}'


def pack(name, flist, out):
    path = os.path.join(out, f"thesis_{name}_{dt.datetime.now():%Y%m%d}.tar" + ('' if name == 'gefs' else '.gz'))
    print(f'写入 {path}（{len(flist)} 个文件，未压缩 {sum(os.path.getsize(os.path.join(ROOT, f)) for f in flist) / 1e9:.2f} GB）…', flush=True)
    manifest = []
    with tarfile.open(path, 'w' if name == 'gefs' else 'w:gz', **({} if name == 'gefs' else {'compresslevel': 6})) as tar:
        for i, f in enumerate(flist, 1):
            p = os.path.join(ROOT, f)
            manifest.append(f'{sha256(p)}  {f}')
            tar.add(p, arcname=f'{TOP}/{f}', recursive=False)
            if i % 2000 == 0:
                print(f'  {i} / {len(flist)}', flush=True)
        if name == 'core':
            manifest += [add_text(tar, k, v) for k, v in environment().items()]
        add_text(tar, f'MANIFEST_{name}.sha256', '\n'.join(manifest) + '\n')
    print(f'完成：{path}，{os.path.getsize(path) / 1e9:.2f} GB，SHA256 {sha256(path)}', flush=True)


def verify(path):
    bad, n = [], 0
    with tarfile.open(path) as tar:
        names = [m for m in tar.getmembers() if os.path.basename(m.name).startswith('MANIFEST_')]
        if len(names) != 1:
            raise SystemExit('包里没有唯一的 MANIFEST')
        want = dict(reversed(ln.split('  ', 1)) for ln in tar.extractfile(names[0]).read().decode('utf-8').splitlines())
        for m in tar:
            if not m.isfile() or m.name == names[0].name:
                continue
            rel = m.name.split('/', 1)[1]
            h = hashlib.sha256()
            for chunk in iter(lambda s=tar.extractfile(m): s.read(1 << 20), b''):
                h.update(chunk)
            n += 1
            if want.pop(rel, None) != h.hexdigest():
                bad.append(rel)
    print(f'{os.path.basename(path)}：核对 {n} 个文件，不一致 {len(bad)} 个，清单里缺 {len(want)} 个。' + ('全部通过' if not bad and not want else ''))
    for f in bad[:20] + list(want)[:20]:
        print('  ', f)
    return not bad and not want


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('parts', nargs='*', help='core、gefs、runs')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify')
    p.add_argument('--out', default=os.path.join(ROOT, 'thesis_pack'))
    a = p.parse_args()
    if set(a.parts) - {'core', 'gefs', 'runs'}:
        raise SystemExit(f'不认识的包：{a.parts}；只能是 core、gefs、runs')
    if a.verify:
        sys.exit(0 if verify(a.verify) else 1)
    g = groups()
    for name, flist in g.items():
        print(f'{name}：{len(flist)} 个文件，未压缩 {sum(os.path.getsize(os.path.join(ROOT, f)) for f in flist) / 1e9:.2f} GB', flush=True)
    missing = [f for f in EXPECT if not os.path.exists(os.path.join(ROOT, f))]
    print('缺失的关键文件：\n  ' + '\n  '.join(missing) if missing else '关键文件都在。', flush=True)
    os.makedirs(a.out, exist_ok=True)
    print(f'{a.out} 所在磁盘剩余 {shutil.disk_usage(a.out).free / 1e9:.0f} GB', flush=True)
    if a.dry_run or not a.parts:
        return
    for name in a.parts:
        pack(name, g[name], a.out)


if __name__ == '__main__':
    main()
