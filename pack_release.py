"""复现包（10-28）：把最终方案的代码、冻结模型、整理好的数据与已报告的结果打成一个 tar.gz，解压到任何目录都能核对。

在训练和评分用的那台机器上运行，因为 GRU 权重和业务检验数据集只在那里：
    python pack_release.py                      # → release/wave_release_<日期>.tar.gz
    python pack_release.py --with-sources       # 另加站点级原始序列（GEFS 回算 / 业务站点抽取、浮标、台风标签），可从头重整数据
做法：
  - 三个冻结运行（Transformer、GRU、周期多任务 Transformer）先在本机核对配置指纹（与冻结时完全一致才打包），
    再把配置里的路径改成相对项目根目录（processed/...、model_runs/...），selection.json 加上与机器无关的
    config_hash_portable——换一台机器、换一个目录解压，verify_release.py 和 run_training.py 仍能确认是同一个冻结配置；
  - 记录环境（Python、pip freeze、torch / CUDA、GPU、系统）到 env/；
  - 每个文件的 SHA256 写进 MANIFEST.sha256（verify_release.py 先核对它）。
原始 GRIB（约 430 GB）不在包内；下载脚本在包内，命令见包里的 README_复现.md。
--origin-root：在冻结机器之外打包（如本机测试）时，给出冻结时的项目根目录，用它核对原始指纹。
"""
import argparse
import datetime as dt
import glob
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from project_paths import resolve_config, LOCAL_ROOTS                        # noqa: E402
from wavecast.experiment import fingerprint, portable_fingerprint, relative_paths  # noqa: E402

RUNS = ['GKH_h15_transformer_pretrain', 'GKH_h15_gru_pretrain', 'GKH_h15_te_transformer_pretrain']
ENSEMBLE = ['GKH_h15_transformer_pretrain', 'GKH_h15_gru_pretrain']
SITES = ['G', 'K', 'H']
OPER_SUFFIXES = ['_oper', '_oper_hsbridge']
TOP = 'wave_release'

CODE = ['*.py', '*.sh', '代码说明.md', '模型训练与验证方案.md', 'metadata/**/*',
        'outputs/wave_forecast/wavecast/*.py', 'outputs/wave_forecast/tests/*.py', 'outputs/wave_forecast/*.md',
        'outputs/wave_forecast/requirements.txt', 'outputs/wave_forecast/config.json']
DATA = [f'processed/site_{s}{x}/**/*' for s in SITES for x in ('_reforecast', '_era5pre', '')] + ['processed/buoy_spectral/**/*']
RESULTS = ['results/postproc/GKH_h15_*', 'results/forecast_outputs/GKH_h15_*', 'results/operational/**/*',
           'results/joint/copula_rho_pit.csv', 'results/*.md']


def files(patterns):
    out = []
    for p in patterns:
        out += [f for f in glob.glob(os.path.join(ROOT, p), recursive=True) if os.path.isfile(f)]
    return sorted({os.path.relpath(f, ROOT).replace('\\', '/') for f in out
                   if '__pycache__' not in f and '_resolved' not in f})


def frozen_check(run, origin_root):
    cfg = json.load(open(resolve_config(os.path.join(ROOT, 'model_configs', f'{run}.json')), encoding='utf-8'))
    sel = json.load(open(os.path.join(cfg['output_dir'], 'selection.json'), encoding='utf-8'))
    ok = sel['config_hash'] == fingerprint(cfg)
    if not ok and origin_root:
        text = json.dumps(cfg)
        for prefix in [ROOT.replace('\\', '/')] + LOCAL_ROOTS:
            text = text.replace(prefix.replace('\\', '/'), origin_root.rstrip('/'))
        ok = sel['config_hash'] == fingerprint(json.loads(text))
    if not ok:
        raise SystemExit(f'{run}：配置与冻结时的指纹不一致，拒绝打包（在冻结的机器上运行，或给出 --origin-root）')
    return cfg, sel


def environment():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
        except Exception as e:
            return f'（无法运行 {cmd[0]}：{e}）'
    info = [f'记录时间 {dt.datetime.now():%Y-%m-%d %H:%M}', f'Python {sys.version}', f'系统 {platform.platform()}']
    try:
        import torch
        info += [f'torch {torch.__version__}，CUDA {torch.version.cuda}，cuDNN {torch.backends.cudnn.version()}，'
                 f'GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无"}']
    except Exception as e:
        info.append(f'torch 不可用：{e}')
    return {'env/environment.txt': '\n'.join(info) + '\n',
            'env/pip_freeze.txt': run([sys.executable, '-m', 'pip', 'freeze']),
            'env/nvidia_smi.txt': run(['nvidia-smi'])}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='release')
    p.add_argument('--with-sources', action='store_true')
    p.add_argument('--origin-root', help='冻结时的项目根目录（只在冻结机器之外打包时用）')
    p.add_argument('--runs', nargs='+', default=RUNS, help='测试用：只打包部分运行')
    a = p.parse_args()
    stamp = dt.datetime.now().strftime('%Y%m%d')

    # 1. frozen runs: exact fingerprint first, then portable copies of config and selection
    gen, frozen = {}, {}
    for run in a.runs:
        cfg, sel = frozen_check(run, a.origin_root)
        sel = dict(sel, config_hash_portable=portable_fingerprint(cfg))
        gen[f'model_runs/{run}/selection.json'] = json.dumps(sel, ensure_ascii=False, indent=2)
        frozen[run] = dict(config_hash=sel['config_hash'], config_hash_portable=sel['config_hash_portable'],
                           best=sel['best'], seeds=cfg['seeds'])
        print(f'[冻结核对通过] {run}：{sel["best"]}', flush=True)
    for f in files(['model_configs/*.json']):
        cfg = json.load(open(os.path.join(ROOT, f), encoding='utf-8'))
        gen[f] = json.dumps(relative_paths(cfg), ensure_ascii=False, indent=1)

    # 2. file list
    groups = {'代码': files(CODE), '模型': files([f'model_runs/{r}/**/*' for r in a.runs]),
              '数据': files(DATA + [f'processed/site_{s}_reforecast{x}/**/*' for s in SITES for x in OPER_SUFFIXES]),
              '结果': files(RESULTS)}
    if a.with_sources:
        stations = []
        meta = {s['id']: s for s in json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']}
        for s in SITES:
            b = (meta.get(s) or {}).get('buoy') or {}
            if b.get('station'):
                stations.append(str(b['station']))
        groups['原始站点序列'] = files([f'processed/gefs_reforecast/{st}/**/*' for st in stations] + ['processed/gefs_operational/**/*']
                                 + [f'processed/buoy/{s}_*.csv' for s in SITES] + [f'processed/tracks/tc_hourly_{s}.csv' for s in SITES])
    for f in gen:
        for g in groups.values():
            if f in g:
                g.remove(f)
    missing = [f'model_runs/{r}/{x}' for r in a.runs for x in ('selection.json', 'config_used.json')
               if not os.path.exists(os.path.join(ROOT, 'model_runs', r, x))]
    missing += [f'processed/site_{s}_reforecast{x}/manifest.json' for s in SITES for x in ['', *OPER_SUFFIXES]
                if not os.path.exists(os.path.join(ROOT, 'processed', f'site_{s}_reforecast{x}', 'manifest.json'))]
    has_test = {r: os.path.exists(os.path.join(ROOT, 'model_runs', r, 'test', 'quantiles.npz')) for r in a.runs}
    oper = [x for x in OPER_SUFFIXES if os.path.exists(os.path.join(ROOT, 'results', 'operational', f'{ENSEMBLE[0]}_ens{x}', 'quantiles.npz'))]
    if missing:
        print('注意，缺少：\n  ' + '\n  '.join(missing), flush=True)

    gen.update(environment())
    gen['release.json'] = json.dumps(dict(created=dt.datetime.now().isoformat(timespec='seconds'), machine=platform.node(),
                                          runs=frozen, ensemble=[r for r in ENSEMBLE if r in a.runs],
                                          test_recorded=has_test, operational_suffixes=oper,
                                          with_sources=a.with_sources, missing=missing), ensure_ascii=False, indent=2)
    readme = os.path.join(ROOT, 'README_复现.md')
    if os.path.exists(readme) and 'README_复现.md' not in groups['代码']:
        groups['代码'].append('README_复现.md')

    # 3. write the archive, hashing as we go
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    out = os.path.join(ROOT, a.out, f'wave_release_{stamp}.tar.gz')
    manifest, total = [], 0
    for name, g in groups.items():
        size = sum(os.path.getsize(os.path.join(ROOT, f)) for f in g)
        total += size
        print(f'  {name}：{len(g)} 个文件，{size / 1e6:.0f} MB', flush=True)
    print(f'写入 {out}（未压缩 {total / 1e9:.2f} GB，压缩需几分钟）…', flush=True)
    with tarfile.open(out, 'w:gz', compresslevel=6) as tar:
        for g in groups.values():
            for f in g:
                path = os.path.join(ROOT, f)
                manifest.append(f'{sha256(path)}  {f}')
                tar.add(path, arcname=f'{TOP}/{f}', recursive=False)
        for f, text in sorted(gen.items()):
            data = text.encode('utf-8')
            manifest.append(f'{hashlib.sha256(data).hexdigest()}  {f}')
            info = tarfile.TarInfo(f'{TOP}/{f}'); info.size = len(data); info.mtime = int(dt.datetime.now().timestamp())
            tar.addfile(info, io.BytesIO(data))
        data = ('\n'.join(sorted(manifest, key=lambda s: s[66:])) + '\n').encode('utf-8')
        info = tarfile.TarInfo(f'{TOP}/MANIFEST.sha256'); info.size = len(data); info.mtime = int(dt.datetime.now().timestamp())
        tar.addfile(info, io.BytesIO(data))
    print(f'完成：{out}，{os.path.getsize(out) / 1e9:.2f} GB，{len(manifest)} 个文件；SHA256 {sha256(out)}', flush=True)
    print(f'核对：mkdir -p ~/repro_check && tar xzf {out} -C ~/repro_check && '
          f'cd ~/repro_check/{TOP} && python verify_release.py', flush=True)


if __name__ == '__main__':
    main()
