"""贝叶斯优化确定损失权重 λ。损失 = CRPS + λ·twCRPS(2.5 m)。

第二版（第一版见 model_runs/GKH_h15_bo_lambda_*，已作废：每次试验只用 1 个种子，总分的高低主要由随机性决定；
采样器又卡在下边界，20 次试验有 15 次重复 λ = 0.05）。这一版的做法：
- 目标：3 个种子（11、23、37）分位数平均后的验证集总分 = CRPS + twCRPS(2.5 m)，与最终产品的做法相同。
- 代理模型：高斯过程（Matérn 5/2 核 + 噪声项，长度尺度下限 1，不去追逐种子噪声），在 log λ 上建模，λ ∈ [0.05, 10]。
- 采集函数：期望提升（EI），每次取 EI 最大的 λ。Optuna 的 GPSampler 在这个目标上一直提边界附近的值，所以不用它。
- 起点：λ 网格已训练好的 0.5、1、2、4（同样的种子和训练流程），不重复训练；之后再做 12 次试验。
- 去重：与已试过的值相差不到约 10% 的 λ 不再选（种子固定时重复训练只会得到相同结果）。
- 每次试验与最终模型同样训练：ERA5 预训练 + 浮标微调 + 按验证 CRPS 早停。只用训练集和验证集，不碰测试集。
- 最优 λ 与 λ = 1 用配对块自助 2000 次比较，规则见 tune_weights.py。
试验记录存在 SQLite 里，中断后重新运行同一条命令会接着做。
输出：model_runs/GKH_h15_bo3_lambda_<模型>/ 下的 trials.csv、best_lambda.json、tune_log.txt、gp_curve.png（后验曲线）。
用法：python tune_lambda_bo.py transformer [--trials 12]
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

LOW, HIGH = 0.05, 10.0
WARM = [(0.5, '05'), (1, '1'), (2, '2'), (4, '4')]
MIN_GAP = 0.1                                   # in log λ, about 10%


def fit_gp(lams, values):
    """Gaussian process of the total score on log λ. Length scale at least 1, so that it does not chase the seed noise."""
    import warnings
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return GaussianProcessRegressor(ConstantKernel(1.0, (1e-2, 1e2)) * Matern(2.0, (1, 10), nu=2.5) + WhiteKernel(.1, (1e-2, 10)),
                                        normalize_y=True, n_restarts_optimizer=5, random_state=0).fit(np.log(lams)[:, None], np.asarray(values))


def next_lambda(lams, values):
    """λ with the largest expected improvement over the best posterior mean at a tried point, at least MIN_GAP from every trial."""
    import numpy as np
    from scipy.stats import norm
    gp = fit_gp(lams, values)
    x, g = np.log(lams), np.linspace(np.log(LOW), np.log(HIGH), 400)
    m, s = gp.predict(g[:, None], return_std=True)
    best = gp.predict(x[:, None]).min()
    z = (best - m) / np.maximum(s, 1e-12)
    ei = (best - m) * norm.cdf(z) + s * norm.pdf(z)
    ei[np.abs(g[:, None] - x[None, :]).min(1) < MIN_GAP] = -1
    return float(np.clip(np.exp(g[int(np.argmax(ei))]), LOW, HIGH))


def gp_curve(trials, out):
    """Posterior of the total score against log λ, from the finished trials."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.log(trials.params_lam.to_numpy())[:, None]
    y = trials.value.to_numpy()
    gp = fit_gp(trials.params_lam.to_numpy(), y)
    g = np.linspace(np.log(LOW), np.log(HIGH), 200)[:, None]
    m, s = gp.predict(g, return_std=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=200)
    ax.fill_between(np.exp(g[:, 0]), m - 1.96 * s, m + 1.96 * s, color='#b7d3f6', alpha=.6, lw=0, label='95% interval')
    ax.plot(np.exp(g[:, 0]), m, color='#2a78d6', lw=1.5, label='GP posterior mean')
    ax.scatter(np.exp(x[:, 0]), y, s=20, color='#22221f', zorder=3, label='Trials (3 seeds)')
    one = trials[np.isclose(trials.params_lam, 1)]
    ax.scatter(one.params_lam, one.value, s=48, color='#eb6834', edgecolor='white', linewidth=.8, zorder=4, label='λ = 1 (current)')
    ax.set_xscale('log'); ax.set_xlabel('Loss weight λ'); ax.set_ylabel('Validation CRPS + twCRPS (m)')
    ax.grid(color='#e8e7e2', lw=.6); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, 'gp_curve.png')); plt.close(fig)
    return float(np.exp(g[int(np.argmin(m)), 0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', choices=['transformer', 'gru'])
    p.add_argument('--trials', type=int, default=12, help='new trials after the four grid points')
    a = p.parse_args()

    config = resolve_config(f'model_configs/GKH_h15_{a.model}_pretrain.json')
    cfg = json.load(open(config, encoding='utf-8'))
    out = os.path.join(os.path.dirname(cfg['output_dir'].rstrip('/\\')), f'GKH_h15_bo3_lambda_{a.model}')
    os.makedirs(out, exist_ok=True)
    sys.stdout = sys.stderr = Tee(sys.__stdout__, open(os.path.join(out, 'tune_log.txt'), 'a', encoding='utf-8', buffering=1))
    print(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} 贝叶斯优化 λ（3 个种子）：{a.model}，网格 {len(WARM)} 点 + 新试验 {a.trials} 次，结果 {out}', flush=True)

    import numpy as np
    import optuna
    import torch
    from optuna.distributions import FloatDistribution
    from optuna.trial import TrialState, create_trial
    from route_models import seed_mean, boot
    from wavecast.data import Scaler, feature_names
    from wavecast.experiment import load_sites, build_all, train_one
    from wavecast.metrics import score

    ablation = cfg['ablations'][-1]
    sites, _ = load_sites(cfg)
    samples, _, _ = build_all(sites, cfg, feature_names(cfg, ablation))
    pre_sites, _ = load_sites(dict(cfg, sites=cfg['pretrain']['sites']), te_proxy=True)
    pre_train = build_all(pre_sites, cfg, feature_names(cfg, ablation), site_ids=[c['site_id'] for c, _, _ in sites])[0]['train']
    scaler = Scaler().fit(samples['train'])
    y = np.concatenate([s['y'] for s in samples['val']])
    blocks = np.array([m['block_id'] for s in samples['val'] for m in s['metadata']])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device'] == 'auto' else cfg['device']
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    print(f'样本：训练 {len(samples["train"])}，预训练 {len(pre_train)}，验证 {len(samples["val"])}；设备 {device}', flush=True)

    def scores(q):
        return score(q, y), score(np.maximum(q, 2.5), np.maximum(y, 2.5))

    dist = {'lam': FloatDistribution(LOW, HIGH, log=True)}
    study = optuna.create_study(direction='minimize', study_name=f'lambda3_{a.model}', storage=f'sqlite:///{os.path.join(out, "study.db")}',
                                load_if_exists=True, sampler=optuna.samplers.RandomSampler(seed=0))      # λ is always enqueued; the study is only the record
    if not study.trials:
        for lam, tag in WARM:
            c, t = scores(seed_mean(f'GKH_h15_lam{tag}_{a.model}_pretrain'))
            study.add_trial(create_trial(params={'lam': float(lam)}, distributions=dist, value=float(c.mean() + t.mean()),
                                         user_attrs=dict(crps=float(c.mean()), twcrps=float(t.mean()), source='网格')))

    def complete():
        return [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]

    while len(complete()) < len(WARM) + a.trials:
        done = complete()
        tried = [t.params['lam'] for t in done]
        study.enqueue_trial({'lam': next_lambda(tried, [t.value for t in done])})
        trial = study.ask(dist)
        lam, source = trial.params['lam'], '期望提升'
        c = dict(cfg, tw_loss=dict(cfg['tw_loss'], weight=lam))
        t0 = time.time()
        print(f'[试验 {len(tried) - len(WARM) + 1}/{a.trials}] λ = {lam:.3f}（{source}）', flush=True)
        q = np.mean([train_one(a.model, s, samples, scaler, c, os.path.join(out, 'trial_tmp.pt'), device, pretrain=pre_train)[0]
                     for s in cfg['seeds']], axis=0)
        np.savez_compressed(os.path.join(out, f'trial{trial.number}_val.npz'), quantiles=q)
        cr, tw = scores(q)
        trial.set_user_attr('crps', float(cr.mean())); trial.set_user_attr('twcrps', float(tw.mean())); trial.set_user_attr('source', source)
        study.tell(trial, float(cr.mean() + tw.mean()))
        print(f'    → 总分 {cr.mean() + tw.mean():.4f}（CRPS {cr.mean():.4f}，twCRPS {tw.mean():.4f}），{time.time() - t0:.0f} s', flush=True)

    trials = study.trials_dataframe()
    trials = trials[trials.state == 'COMPLETE'].sort_values('params_lam')
    trials.to_csv(os.path.join(out, 'trials.csv'), index=False)
    gp_best = gp_curve(trials, out)
    best = study.best_trial
    tag = dict((float(l), g) for l, g in WARM).get(best.params['lam'])
    qb = seed_mean(f'GKH_h15_lam{tag}_{a.model}_pretrain') if tag else np.load(os.path.join(out, f'trial{best.number}_val.npz'))['quantiles']
    s1, sb = (lambda c: c[0] + c[1])(scores(seed_mean(f'GKH_h15_lam1_{a.model}_pretrain'))), (lambda c: c[0] + c[1])(scores(qb))
    d, lo, hi = boot(s1, sb, blocks, np.random.default_rng(0))
    json.dump(dict(model=a.model, best_trial_lambda=best.params['lam'], best_trial_score=best.value, gp_posterior_min_lambda=gp_best,
                   relative_to_lambda1=d, ci95=[lo, hi]),
              open(os.path.join(out, 'best_lambda.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'=== 试验中最好 λ = {best.params["lam"]:.3f}（总分 {best.value:.4f}），相对 λ = 1：{d:+.2%} [{lo:+.2%}, {hi:+.2%}]；'
          f'高斯过程后验最小处 λ = {gp_best:.3f}', flush=True)


if __name__ == '__main__':
    main()
