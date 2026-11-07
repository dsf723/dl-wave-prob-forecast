"""贝叶斯超参数调优（Optuna TPE，参考 Xia et al. 2026, Ocean Eng. 368 用贝叶斯优化自动调参）。

只用训练集训练、验证集打分（目标与正式试验相同：0.5 × 总体 CRPS + 0.5 × 极端天气 CRPS），**不碰测试集**。
搜索：hidden {32, 64, 128}、layers {1, 2, 3}、dropout [0, 0.3]、learning_rate [1e-4, 3e-3]（对数）、
      batch_size {16, 32, 64}、degree {8, 12, 16}（Bernstein 阶数）；Transformer / CNN-BiGRU 还调 heads {2, 4, 8}。
每次试验用 1 个种子、较少轮数（--epochs，默认 30；早停 --patience，默认 5），所以比正式训练快很多。
试验记录存在 SQLite 数据库里，中断后重新运行同一条命令会接着做（Optuna 的 load_if_exists）。
输出：<原 output_dir>_tune_<模型>/ 下的 trials.csv、best_params.json、tune_log.txt；
      并生成调好参数的新配置 model_configs/<原配置名>_tuned_<模型>.json（models 只含该模型、output_dir 是新目录），
      之后用 run_training.py 正式跑这个新配置（3 个种子、完整轮数）。
需要：pip install optuna
用法：python tune_hyperparams.py model_configs/GKH_reforecast_joint_tw.json --model transformer --trials 30
      其他：--ablation season（默认取配置 ablations 的最后一个）、--epochs 30、--patience 5、--seed 11
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, 'outputs', 'wave_forecast')
sys.path.insert(0, ROOT)
sys.path.insert(0, PKG)

from project_paths import resolve_config       # noqa: E402
from run_training import Tee                    # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--model', default='transformer', choices=['tide', 'transformer', 'gru', 'cnn_bigru'])
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--ablation')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=11)
    a = p.parse_args()

    config = resolve_config(a.config)
    cfg = json.load(open(config, encoding='utf-8'))
    ablation = a.ablation or cfg['ablations'][-1]
    out = cfg['output_dir'].rstrip('/\\') + f'_tune_{a.model}'
    os.makedirs(out, exist_ok=True)
    sys.stdout = sys.stderr = Tee(sys.__stdout__, open(os.path.join(out, 'tune_log.txt'), 'a', encoding='utf-8', buffering=1))
    print(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} 贝叶斯调参：{a.model}，输入组 {ablation}，{a.trials} 次试验，结果 {out}', flush=True)

    import numpy as np
    import optuna
    import torch
    from wavecast.data import Scaler, feature_names
    from wavecast.experiment import load_sites, build_all, train_one, climatology, objective
    from wavecast.metrics import case_table

    sites, _ = load_sites(cfg)
    samples, _, _ = build_all(sites, cfg, feature_names(cfg, ablation))
    scaler = Scaler().fit(samples['train'])
    clim = climatology(sites, samples['val'], cfg['target'])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device'] == 'auto' else cfg['device']
    print(f'样本：训练 {len(samples["train"])}，验证 {len(samples["val"])}；设备 {device}', flush=True)

    def trial_fn(trial):
        params = dict(hidden=trial.suggest_categorical('hidden', [32, 64, 128]),
                      layers=trial.suggest_int('layers', 1, 3),
                      dropout=trial.suggest_float('dropout', 0.0, 0.3),
                      learning_rate=trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
                      batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
                      degree=trial.suggest_categorical('degree', [8, 12, 16]))
        if a.model in ('transformer', 'cnn_bigru'):
            params['heads'] = trial.suggest_categorical('heads', [2, 4, 8])
        c = dict(cfg, **params, epochs=a.epochs, patience=a.patience)
        t0 = time.time()
        print(f'[试验 {trial.number + 1}/{a.trials}] {params}', flush=True)
        q, history, n_params = train_one(a.model, a.seed, samples, scaler, c, os.path.join(out, 'trial_tmp.pt'), device)
        obj = objective(case_table(samples['val'], q, f'{a.model}_trial', c, clim))
        trial.set_user_attr('n_params', int(n_params))
        trial.set_user_attr('epochs_run', len(history))
        print(f'    → 验证目标 {obj:.4f}，参数量 {n_params}，{len(history)} 轮，{time.time() - t0:.0f} s', flush=True)
        return obj

    study = optuna.create_study(direction='minimize', study_name=f'{a.model}_{ablation}',
                                storage=f'sqlite:///{os.path.join(out, "study.db")}', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0))
    remaining = a.trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if remaining > 0:
        study.optimize(trial_fn, n_trials=remaining)
    study.trials_dataframe().to_csv(os.path.join(out, 'trials.csv'), index=False)
    best = study.best_trial
    json.dump(dict(model=a.model, ablation=ablation, objective=best.value, params=best.params,
                   note='Chosen on validation with 1 seed and short training; confirm with a full run_training.py run.'),
              open(os.path.join(out, 'best_params.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    tuned = json.load(open(os.path.abspath(a.config), encoding='utf-8'))      # keep the user's original paths
    tuned.update(best.params, models=[a.model], output_dir=tuned['output_dir'].rstrip('/\\') + f'_tuned_{a.model}')
    stem = os.path.splitext(os.path.basename(a.config))[0]
    tuned_path = os.path.join(ROOT, 'model_configs', f'{stem}_tuned_{a.model}.json')
    json.dump(tuned, open(tuned_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'=== 最优验证目标 {best.value:.4f}，参数 {best.params}', flush=True)
    print(f'    新配置：{tuned_path}', flush=True)
    print(f'    正式训练：python run_training.py model_configs/{stem}_tuned_{a.model}.json validate', flush=True)


if __name__ == '__main__':
    main()
