"""打包上传到云服务器的文件：server_upload/wave_server.zip（解压后是 wave_project/ 目录）。

包含：模型代码（wavecast）、训练 / 调参 / 门控 / GEFS 回算下载脚本、全部配置、G/K/H 三站真实回算数据集、训练手册。
不含：demo 数据、A 站伪预报数据、训练结果、预训练数据集（需要时再加）。
用法：python pack_server.py
      python pack_server.py --code-only   # 10-28：只打代码、配置和结果说明（server_upload/wave_code.zip，几 MB），数据已在服务器上时用
"""
import glob
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = ['run_training.py', 'run_tc_gate.py', 'project_paths.py', 'tune_hyperparams.py', 'download_gefs_reforecast.py', 'download_tigge_tracks.py', 'parse_tigge_tracks.py', 'compare_runs.py', 'site_skill.py', 'joint_scenarios.py', 'merge_regional_exports.py', 'site_model_choice.py', 'selection_rule_check.py', 'postproc_diagnostics.py', 'tail_fix.py', 'compare_tail_variants.py', 'copula_from_pit.py', 'route_models.py', 'energy_check.py', 'forecast_monthly_energy.py', 'forecast_wave_energy_mvi.py', 'tigge_track_features.py', 'prepare_reforecast_dataset.py', 'check_reforecast_files.py', 'metadata/sites.json', 'download_gefs_operational.py', 'metadata/gefs_reforecast_era5_ratios.json', 'evaluate_operational.py', 'run_oper_download.sh', 'results/joint/copula_rho_pit.csv',
         'train_extreme_specialist.py', 'export_quantile_forecasts.py', '模型训练与验证方案.md',
         'outputs/wave_forecast/requirements.txt', 'outputs/wave_forecast/README.md',
         'outputs/wave_forecast/DATA_SCHEMA.md', 'outputs/wave_forecast/config.json',
         'outputs/wave_forecast/tests/test_core.py', 'outputs/wave_forecast/tests/test_models_extra.py', 'outputs/wave_forecast/tests/test_postproc.py']
DATA = ['site_G_reforecast', 'site_K_reforecast', 'site_H_reforecast', 'site_H2_reforecast',
        'site_G_era5pre', 'site_K_era5pre', 'site_H_era5pre',          # 10-22：加 H2 与预训练数据
        'site_G_reforecast_tc', 'site_K_reforecast_tc', 'site_H_reforecast_tc']   # 10-22：TIGGE 台风路径特征


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    os.chdir(ROOT)
    items = FILES + glob.glob('outputs/wave_forecast/wavecast/*.py') + glob.glob('model_configs/*.json')
    if '--code-only' in sys.argv:
        # 10-28 reproduction package: every top-level script (pack_release.py takes them all) plus the result notes
        # written on this machine (results/*.md, operational extremes)
        items = sorted(set(f for f in items if f.endswith(('.py', '.sh', '.md', '.json', '.txt', '.csv')) and not f.startswith('processed/'))
                       | set(glob.glob('*.py') + glob.glob('*.sh') + ['README_复现.md', '代码说明.md']
                             + glob.glob('results/*.md') + glob.glob('results/operational/*_extremes.md')))
        out = os.path.join('server_upload', 'wave_code.zip')
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in items:
                z.write(f, 'wave_project/' + f.replace(os.sep, '/'))
        print(f'{len(items)} 个文件 → {os.path.getsize(out) / 1e6:.1f} MB → {out}')
        return
    for d in DATA:
        items += glob.glob(f'processed/{d}/*')
    # 10-23：在服务器上整理平日回算所需的输入（浮标真值、ERA5 过去 7 天输入、TIGGE 路径表与起报清单）
    kr = ['KR22101', 'KR22102', 'KR22103', 'KR22104', 'KR22105', 'KR22106', 'KR22108', 'KR22185', 'KR22186',
          'KR22188', 'KR22189', 'KR22190', 'KR21229']                       # 10-24：韩国长记录站（服务器上建回算数据集）
    for sid in ('G', 'K', 'H', *kr):
        obs = f'processed/site_{sid}/observations.csv'
        items += glob.glob(f'processed/buoy/{sid}_*.csv') + ([obs] if os.path.exists(obs) else [])
    items += ['tigge_tracks/ecmf_tracks_pacific.csv.gz', 'tigge_tracks/available_runs.txt']
    items += glob.glob('processed/buoy_spectral/*.csv')          # 10-23：CDIP 实测 Te（周期订正的真值）
    os.makedirs('server_upload', exist_ok=True)
    out = os.path.join('server_upload', 'wave_server.zip')
    n = raw = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(items):
            z.write(f, 'wave_project/' + f.replace(os.sep, '/'))
            n += 1
            raw += os.path.getsize(f)
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
    print(f'{n} 个文件，{raw / 1e6:.0f} MB → {os.path.getsize(out) / 1e6:.1f} MB，{"压缩包完好" if bad is None else "损坏：" + bad} → {out}')


if __name__ == '__main__':
    main()
