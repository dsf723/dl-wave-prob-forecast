"""风场试验自动流水线（A 站，S3 第 4 组输入）：等 ERA5 区域风场下完 → 算上游扇区风特征 → 生成伪预报 → 训练验证。

伪预报设定与敏感性试验相同（误差倍率 1、浪–风误差相关 0.95），消融 weather（点风）与 windfield（点风 + 风场），
两者在同一数据集、同一批验证样本上比较，回答"风场比点风多带来多少技巧"。
用法：pythonw pipeline_windfield_A.py      （计划任务"风浪研究-风场试验流水线"）
日志：model_runs/site_A_E2_scale1_rho0.95_wf/pipeline_log.txt，训练日志 train_log_validate.txt
"""
import json
import os
import runpy
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, 'outputs', 'wave_forecast')
NAME = 'site_A_E2_scale1_rho0.95_wf'
DATA = os.path.join(ROOT, 'processed', NAME)
OUT = os.path.join(ROOT, 'model_runs', NAME)
CONFIG = os.path.join(ROOT, 'model_configs', 'A_E2_scale1_rho0.95_wf.json')


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S ') + msg, flush=True)


def main():
    import pandas as pd
    need = [os.path.join(ROOT, 'era5_data', f'windfield_EA_{y}.nc') for y in range(1995, 2025)]
    for _ in range(288):                                    # 最多等 2 天
        missing = [p for p in need if not os.path.exists(p)]
        if not missing:
            break
        log(f'等待风场下载：还缺 {len(missing)} 个年份')
        time.sleep(600)
    else:
        raise SystemExit('风场 2 天内未下完，退出')
    sys.path.insert(0, ROOT)
    import wind_features
    wind_features.run('A')
    obs = pd.read_csv(os.path.join(ROOT, 'processed', 'site_A', 'observations.csv'))
    wf = pd.read_csv(os.path.join(ROOT, 'processed', 'site_A', 'windfield_features.csv'))
    merged = obs.merge(wf, on='valid_time', how='left')
    log(f'合并风场特征：{wf.shape[1] - 1} 列，缺测比例 {merged.filter(like="wf_").isna().mean().mean():.3%}')
    src = os.path.join(ROOT, 'processed', 'site_A', 'observations_with_windfield.csv')
    merged.to_csv(src, index=False)
    sys.path.insert(0, PKG)
    from wavecast.pseudo import generate
    if not os.path.exists(os.path.join(DATA, 'manifest.json')):
        generate(src, DATA, error_scale=1, wind_wave_corr=0.95)
        log(f'伪预报生成完成：{DATA}')
    cfg = json.load(open(os.path.join(ROOT, 'model_configs', 'A_E2_scale1_rho0.95.json'), encoding='utf-8'))
    cfg['features']['windfield'] = [c for c in wf.columns if c.startswith('wf_')]
    cfg.update(data_dir=DATA.replace('\\', '/'), output_dir=OUT.replace('\\', '/'), ablations=['weather', 'windfield'])
    json.dump(cfg, open(CONFIG, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log(f'配置已写：{CONFIG}，开始训练')
    train_log = open(os.path.join(OUT, 'train_log_validate.txt'), 'a', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = train_log
    os.chdir(PKG)
    sys.argv = ['wavecast.experiment', '--config', CONFIG, '--stage', 'validate']
    runpy.run_module('wavecast.experiment', run_name='__main__')
    print(time.strftime('=== %Y-%m-%d %H:%M:%S 完成'), flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    sys.stdout = sys.stderr = open(os.path.join(OUT, 'pipeline_log.txt'), 'a', encoding='utf-8', buffering=1)
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
