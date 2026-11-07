"""用方法确定两个权重，只用验证集：
  1. 损失权重 λ：损失 = CRPS + λ·twCRPS(2.5 m)。λ ∈ {0, 0.5, 1, 2, 4} 各训练一次（3 个种子），
     比较验证集总分 = CRPS + twCRPS（评价口径与论文一致）。
  2. 集成权重 w：分位数 q = w·q_Transformer + (1 − w)·q_GRU，w 从 0 到 1 每 0.05 一格，取验证集总分最小的 w；
     并用分块交叉验证（按站点 × 起报年月分 5 折，4 折选 w、1 折评分）检验"选出来的 w"在没见过的数据上是否仍好于 0.5，
     用块自助 2000 次给出最优 w 的 95% 区间。
事先定的规则（结果出来前写定）：取验证集总分最小的值；若它与现用值（λ = 1、w = 0.5）的差异
不显著（配对块自助 95% 区间含 0），说明现用值已在最优附近，保留现用值，不重新动测试集。
用法：
  python tune_weights.py lambda                       # 汇总 λ 网格（Transformer、GRU 各一张表）
  python tune_weights.py ensemble [--t RUN] [--g RUN] # 集成权重，默认用本机 λ=1 的两次运行
  python tune_weights.py pareto                       # 帕累托前沿：不合成总分，同时看 CRPS 与 twCRPS（网格 + 贝叶斯优化的试验）
  python tune_weights.py pca                          # PCA 与基准比较精度和训练用时（单独运行的 GKH_h15_solo_*）
  python tune_weights.py sweep                        # 主成分数从多到少：13 个变量（11→1）与 11 个集合成员（11→1），表和图
"""
import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from route_models import seed_mean, boot                                     # noqa: E402
from wavecast.metrics import score                                           # noqa: E402

LAMBDAS = [(0, '0'), (0.5, '05'), (1, '1'), (2, '2'), (4, '4')]
OUT = os.path.join(ROOT, 'results', 'weights')


def tw(q, y, t=2.5):
    return score(np.maximum(q, t), np.maximum(y, t))


def total(q, y):
    return score(q, y) + tw(q, y)


def cases(run):
    """Validation truth and blocks in the row order of the run's *_val.npz."""
    import json
    best = json.load(open(os.path.join(ROOT, 'model_runs', run, 'selection.json'), encoding='utf-8'))['best']
    model, info = next(iter(best.items()))
    cs = pd.read_csv(os.path.join(ROOT, 'model_runs', run, 'validation', 'case_scores.csv'),
                     usecols=['model', 'truth', 'block_id', 'site_id', 'issue_time', 'lead_day'])
    return cs[cs.model == f"{model}_{info['ablation']}"].reset_index(drop=True)


def train_minutes(run):
    """Wall time of the last complete validate run, from the start / finish lines of its log."""
    log = os.path.join(ROOT, 'model_runs', run, 'train_log_validate.txt')
    if not os.path.exists(log):
        return np.nan
    t0 = minutes = None
    for line in open(log, encoding='utf-8'):
        m = re.match(r'=== (\S+ \S+) (开始|完成)', line)
        if m:
            t = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            if m.group(2) == '开始':
                t0 = t
            elif t0:
                minutes = (t - t0).total_seconds() / 60
    return minutes if minutes is not None else np.nan


def verdict(lo, hi):
    return '显著变好' if hi < 0 else ('显著变差' if lo > 0 else '不显著')


def table(df):
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    lines += ['| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r) + ' |' for _, r in df.iterrows()]
    return lines


def do_lambda():
    lines = ['# 损失权重 λ 的网格搜索（验证集）', '',
             '损失 = CRPS + λ·twCRPS(2.5 m)；每个 λ 训练 3 个种子取平均，其余设置与最终模型相同。总分 = CRPS + twCRPS（评价口径固定）。'
             '变化 = 相对 λ = 1 的总分变化，方括号为按 block_id 配对块自助 2000 次的 95% 区间。', '']
    rng = np.random.default_rng(0)
    choice = {}
    for m in ['transformer', 'gru']:
        runs = {lam: f'GKH_h15_lam{tag}_{m}_pretrain' for lam, tag in LAMBDAS}
        runs = {lam: r for lam, r in runs.items() if os.path.exists(os.path.join(ROOT, 'model_runs', r, 'selection.json'))}
        if 1 not in runs:
            lines += [f'## {m}', '', 'λ = 1 还没训练完，跳过。', '']
            continue
        cs = cases(runs[1])
        y, blocks = cs.truth.to_numpy(float), cs.block_id.to_numpy()
        base = total(seed_mean(runs[1]), y)
        rows = []
        for lam, r in runs.items():
            q = seed_mean(r)
            if len(q) != len(y):
                raise SystemExit(f'{r} 的验证集行数与 λ=1 不同')
            s = total(q, y)
            d, lo, hi = boot(base, s, blocks, rng)
            rows.append(dict(λ=lam, CRPS=score(q, y).mean(), twCRPS=tw(q, y).mean(), 总分=s.mean(),
                             相对λ1=f'{d:+.2%} [{lo:+.2%}, {hi:+.2%}]', 判断='基准' if lam == 1 else verdict(lo, hi),
                             训练分钟=train_minutes(r)))
        res = pd.DataFrame(rows).sort_values('λ')
        best = res.loc[res['总分'].idxmin()]
        keep = best['λ'] == 1 or best['判断'] != '显著变好'
        choice[m] = 1.0 if keep else float(best['λ'])
        lines += [f'## {m}', ''] + table(res) + ['',
                  f'总分最小的是 λ = {best["λ"]:g}（{best["总分"]:.4f}）；' +
                  ('它就是现用值。' if best['λ'] == 1 else f'相对 λ = 1 {best["相对λ1"]}，{best["判断"]}。') +
                  f'按事先规则取 λ = {choice[m]:g}。', '']
    return lines, choice


def do_ensemble(t_run, g_run, step=.05, folds=5, n_boot=2000):
    cs = cases(t_run)
    y = cs.truth.to_numpy(float)
    qt, qg = seed_mean(t_run), seed_mean(g_run)
    if not (len(qt) == len(qg) == len(y)):
        raise SystemExit('两个运行的验证集行数不一致')
    ws = np.round(np.arange(0, 1 + 1e-9, step), 2)
    # per-row total score for each weight (rows × weights)
    S = np.stack([total(w * qt + (1 - w) * qg, y) for w in ws], axis=1)
    blocks = cs.block_id.to_numpy()
    bsum = pd.DataFrame(S).groupby(blocks).sum()            # block sums: blocks × weights
    nb = pd.Series(1, index=blocks).groupby(level=0).sum().reindex(bsum.index).to_numpy()
    mean = S.mean(0)
    i_best, i_half = int(np.argmin(mean)), int(np.where(ws == .5)[0][0])
    rng = np.random.default_rng(0)
    d, lo, hi = boot(S[:, i_half], S[:, i_best], blocks, rng, n_boot)
    # bootstrap distribution of the optimal weight
    idx = rng.integers(0, len(bsum), (n_boot, len(bsum)))
    wb = ws[np.argmin(bsum.to_numpy()[idx].sum(1), axis=1)]
    # blocked cross-validation: folds of site × issue month
    key = cs.site_id + '_' + cs.issue_time.str[:7]
    uk = np.array(sorted(key.unique()))
    fold_of = dict(zip(uk, rng.permutation(len(uk)) % folds))
    f = key.map(fold_of).to_numpy()
    cv_chosen, cv_score, half_score = [], np.empty(len(y)), np.empty(len(y))
    for k in range(folds):
        tr, te = f != k, f == k
        j = int(np.argmin(S[tr].mean(0)))
        cv_chosen.append(ws[j]); cv_score[te] = S[te, j]; half_score[te] = S[te, i_half]
    dcv, locv, hicv = boot(half_score, cv_score, blocks, rng, n_boot)
    # lead-dependent weights (D1-3 / D4-7 / D8-15), for information
    band = np.where(cs.lead_day <= 3, 'D1-3', np.where(cs.lead_day <= 7, 'D4-7', 'D8-15'))
    by_band = {b: ws[int(np.argmin(S[band == b].mean(0)))] for b in ['D1-3', 'D4-7', 'D8-15']}

    lines = ['# 集成权重 w 的确定（验证集）', '',
             f'q = w·q_Transformer + (1 − w)·q_GRU；Transformer = `{t_run}`，GRU = `{g_run}`，各 3 个种子平均；'
             f'验证集 {len(y)} 个时刻。总分 = CRPS + twCRPS(2.5 m)。', '',
             '| w（Transformer 的权重） | CRPS | twCRPS | 总分 |', '|---:|---:|---:|---:|']
    for j, w in enumerate(ws):
        q = w * qt + (1 - w) * qg
        mark = ' ←最小' if j == i_best else (' ←现用' if j == i_half else '')
        lines.append(f'| {w:.2f}{mark} | {score(q, y).mean():.4f} | {tw(q, y).mean():.4f} | {mean[j]:.4f} |')
    lines += ['',
              f'- 验证集总分最小：w = {ws[i_best]:.2f}（{mean[i_best]:.4f}）；现用 w = 0.50（{mean[i_half]:.4f}）；'
              f'最小值相对 0.5：{d:+.2%} [{lo:+.2%}, {hi:+.2%}]，{verdict(lo, hi)}。',
              f'- 块自助 {n_boot} 次的最优 w：中位数 {np.median(wb):.2f}，95% 区间 [{np.percentile(wb, 2.5):.2f}, {np.percentile(wb, 97.5):.2f}]。',
              f'- 分块交叉验证（{folds} 折，按站点 × 起报年月）：各折选出的 w = {", ".join(f"{w:.2f}" for w in cv_chosen)}；'
              f'用"各折选出的 w"在留出折上评分，相对固定 0.5：{dcv:+.2%} [{locv:+.2%}, {hicv:+.2%}]，{verdict(locv, hicv)}。',
              f'- 分时效各自最优（仅供参考）：' + '，'.join(f'{b} w = {w:.2f}' for b, w in by_band.items()) + '。']
    keep = i_best == i_half or verdict(lo, hi) != '显著变好'
    w_star = .5 if keep else float(ws[i_best])
    lines.append(f'- 按事先规则取 w = {w_star:.2f}。')
    return lines, w_star


def points(m):
    """All λ tried for one model, each the 3-seed quantile mean as in the product (grid runs + Bayesian-optimisation trials)."""
    rows = []
    base = f'GKH_h15_lam1_{m}_pretrain'
    if os.path.exists(os.path.join(ROOT, 'model_runs', base, 'selection.json')):
        y = cases(base).truth.to_numpy(float)
        for lam, tag in LAMBDAS:
            run = f'GKH_h15_lam{tag}_{m}_pretrain'
            if os.path.exists(os.path.join(ROOT, 'model_runs', run, 'selection.json')):
                q = seed_mean(run)
                rows.append(dict(λ=float(lam), CRPS=score(q, y).mean(), twCRPS=tw(q, y).mean(), 来源='网格'))
    db = os.path.join(ROOT, 'model_runs', f'GKH_h15_bo3_lambda_{m}', 'study.db')
    if os.path.exists(db):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for t in optuna.load_study(study_name=f'lambda3_{m}', storage=f'sqlite:///{db}').get_trials(deepcopy=False):
            if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get('source') != '网格':
                rows.append(dict(λ=t.params['lam'], CRPS=t.user_attrs['crps'], twCRPS=t.user_attrs['twcrps'], 来源='贝叶斯优化'))
    return pd.DataFrame(rows)


def on_front(df):
    """True where no other point is at least as good on both scores and strictly better on one."""
    c, t = df.CRPS.to_numpy(), df.twCRPS.to_numpy()
    return np.array([not (((c <= c[i]) & (t <= t[i]) & ((c < c[i]) | (t < t[i]))).any()) for i in range(len(df))])


def seed_noise(m):
    """Standard error of the 3-seed mean of the two scores at λ = 1 (the yardstick for 'how close to the front')."""
    run = f'GKH_h15_lam1_{m}_pretrain'
    y = cases(run).truth.to_numpy(float)
    qs = [np.load(os.path.join(ROOT, 'model_runs', run, f'{m}_season_seed{s}_val.npz'))['quantiles'] for s in (11, 23, 37)]
    return (float(np.std([score(q, y).mean() for q in qs], ddof=1)) / 3 ** .5,
            float(np.std([tw(q, y).mean() for q in qs], ddof=1)) / 3 ** .5)


def do_pareto():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FRONT, BASE, DOM, INK, MUTED = '#2a78d6', '#eb6834', '#b8b7b0', '#22221f', '#6f6e68'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
    lines = ['# 损失权重 λ 的帕累托前沿（验证集）', '',
             '不把两项合成总分：一个 λ 在前沿上，指没有别的 λ 能同时让 CRPS 和 twCRPS 都不变差、且至少一项更好。'
             '点来自 λ 网格（含 λ = 0）与贝叶斯优化的全部试验，每个点都是 3 个种子分位数平均后的成绩，与产品一致。'
             '"噪声尺度"是两个点之差的标准误（λ = 1 时种子间标准差 ÷ √3 × √2），差距小于它就分不出好坏。', '']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), dpi=200)
    for ax, m in zip(axes, ['transformer', 'gru']):
        df = points(m)
        if df.empty:
            ax.set_axis_off(); lines += [f'## {m}', '', '还没有结果。', '']
            continue
        df = df.sort_values('λ').reset_index(drop=True)
        df['前沿'] = on_front(df)
        f = df[df['前沿']].sort_values('CRPS')
        d = df[~df['前沿']]
        ax.scatter(d.CRPS, d.twCRPS, s=22, color=DOM, label='Dominated', zorder=2)
        ax.plot(f.CRPS, f.twCRPS, color=FRONT, lw=1.5, zorder=3)
        ax.scatter(f.CRPS, f.twCRPS, s=30, color=FRONT, edgecolor='white', linewidth=.8, label='Pareto front', zorder=4)
        # label the two ends of the front only: its middle points sit too close together to read
        for _, r in (f.iloc[[0, -1]] if len(f) > 1 else f).iterrows():
            ax.annotate(f'λ = {r["λ"]:.2g}', (r.CRPS, r.twCRPS), xytext=(5, 5), textcoords='offset points', fontsize=8, color=INK)
        one = df[np.isclose(df['λ'], 1)]
        ax.scatter(one.CRPS, one.twCRPS, s=48, color=BASE, edgecolor='white', linewidth=.8, label='λ = 1 (current)', zorder=5)
        ax.set_title(m.upper() if m == 'gru' else m.capitalize(), loc='left', fontsize=10, color=INK)
        ax.set_xlabel('Validation CRPS (m)'); ax.set_ylabel('Validation twCRPS, 2.5 m (m)')
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(color='#e8e7e2', lw=.6); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        lines += [f'## {m}', '', f'共 {len(df)} 个点，前沿上 {len(f)} 个。', '']
        show = df[['λ', 'CRPS', 'twCRPS', '来源', '前沿']].copy()
        show['λ'] = show['λ'].map(lambda v: f'{v:.3f}')
        show['前沿'] = show['前沿'].map(lambda v: '是' if v else '')
        lines += table(show) + ['']
        if len(one):
            nc, nt = seed_noise(m)
            b = one.iloc[0]
            better = df[(df.CRPS <= b.CRPS) & (df.twCRPS <= b.twCRPS) & ((df.CRPS < b.CRPS) | (df.twCRPS < b.twCRPS))]
            nc, nt = nc * 2 ** .5, nt * 2 ** .5
            lines.append(f'噪声尺度：CRPS {nc:.4f}，twCRPS {nt:.4f}。')
            if better.empty:
                lines.append('λ = 1 在前沿上。')
            else:
                gc, gt = (b.CRPS - better.CRPS).max(), (b.twCRPS - better.twCRPS).max()
                lines.append(f'λ = 1 不在前沿上：有 {len(better)} 个点同时不差于它，最大领先 CRPS {gc:.4f}、twCRPS {gt:.4f}（'
                             + ('都在噪声尺度以内，分不出好坏' if gc <= nc and gt <= nt else '超过噪声尺度') + '）。')
            lines.append('')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'pareto_lambda.png'))
    plt.close(fig)
    return lines


def run_stats(run):
    """Model inputs, parameters, epochs trained (pretraining + fine-tuning, all seeds) and summed epoch seconds, from the log."""
    log = open(os.path.join(ROOT, 'model_runs', run, 'train_log_validate.txt'), encoding='utf-8', errors='replace').read()
    start = log.rfind('开始 validate')
    log = log[start:]
    secs = re.findall(r'(?:pretrain \d+/\d+|epoch \d+/\d+):.*\((\d+) s\)', log)
    pca = re.search(r'model inputs (\d+)', log)
    sr = pd.read_csv(os.path.join(ROOT, 'model_runs', run, 'seed_results.csv'))
    return int(pca.group(1)) if pca else 20, int(sr.parameters.iloc[0]), len(secs), sum(map(int, secs))


def do_pca():
    lines = ['# PCA 降维与基准的比较（验证集）', '',
             'PCA 只在训练集上拟合，站点编号不参与。总分 = CRPS + twCRPS(2.5 m)，3 个种子分位数平均。变化 = 相对不用 PCA 的基准，方括号为配对块自助 2000 次的 95% 区间。'
             '用时都是机器上没有其他训练时单独运行的结果；"训练秒数"是预训练加微调各轮用时之和（日志按整秒记），"总分钟"还包括读数据。', '']
    rng = np.random.default_rng(0)
    for m in ['transformer', 'gru']:
        base = f'GKH_h15_solo_base_{m}_pretrain'
        cs = cases(base)
        y, blocks = cs.truth.to_numpy(float), cs.block_id.to_numpy()
        s0 = total(seed_mean(base), y)
        rows = []
        variants = [('基准（7 个输入）', base), ('PCA 95% 方差（6 个主成分）', f'GKH_h15_solo_pca95_{m}_pretrain'),
                    ('PCA 4 个主成分（77% 方差）', f'GKH_h15_solo_pcak4_{m}_pretrain')]
        if m == 'transformer':
            variants.append(('7 个输入 + 过去的天气 6 个 → PCA 95%（11 个主成分）', 'GKH_h15_pcaW95_transformer_pretrain'))
        for name, run in variants:
            if not os.path.exists(os.path.join(ROOT, 'model_runs', run, 'selection.json')):
                rows.append({'方案': name, '总分': '未完成'})
                continue
            q = seed_mean(run)
            s = total(q, y)
            d, lo, hi = boot(s0, s, blocks, rng)
            n_in, n_par, n_ep, sec = run_stats(run)
            rows.append({'方案': name, '模型输入列': n_in, '参数量': n_par, 'CRPS': score(q, y).mean(), 'twCRPS': tw(q, y).mean(), '总分': s.mean(),
                         '相对基准': '—' if run == base else f'{d:+.2%} [{lo:+.2%}, {hi:+.2%}]', '判断': '—' if run == base else verdict(lo, hi),
                         '总轮数': n_ep, '训练秒数': sec, '每轮秒数': sec / max(n_ep, 1), '总分钟': train_minutes(run)})
        lines += [f'## {m}', ''] + table(pd.DataFrame(rows)) + ['']
    return lines


def do_sweep():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    VARS, MEMB, BASE, INK, MUTED = '#2a78d6', '#eb6834', '#6f6e68', '#22221f', '#6f6e68'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
    lines = ['# 主成分数从多到少（验证集）', '',
             '两组：①13 个变量（最终 7 个输入 + 过去的 6 个天气变量）的 PCA；②11 个集合成员波高的 PCA，主成分代替集合平均和离散度，其余输入不变。'
             '变化 = 相对最终 7 个输入（不用 PCA）的总分，方括号为配对块自助 2000 次的 95% 区间。3 个种子分位数平均。'
             '用时是几条训练同时运行时的结果，只有每轮秒数之间大致可比。', '']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, ['transformer', 'gru']):
        base = f'GKH_h15_solo_base_{m}_pretrain'
        cs = cases(base)
        y, blocks = cs.truth.to_numpy(float), cs.block_id.to_numpy()
        s0 = total(seed_mean(base), y)
        groups = [('13 个变量', VARS, 'o', [('不用 PCA', 13, f'GKH_h15_sweep_raw13_{m}_pretrain')] +
                   [(f'{k} 个主成分', k, 'GKH_h15_pcaW95_transformer_pretrain' if (k, m) == (11, 'transformer') else f'GKH_h15_sweep_k{k}_{m}_pretrain')
                    for k in range(11, 0, -1)]),
                  ('11 个成员', MEMB, 's', [(f'{k} 个主成分', k, f'GKH_h15_mem_k{k}_{m}_pretrain') for k in (11, 8, 5, 3, 2, 1)])]
        rows = []
        for gname, color, marker, runs in groups:
            xs, ys = [], []
            for name, k, run in runs:
                if not os.path.exists(os.path.join(ROOT, 'model_runs', run, 'selection.json')):
                    continue
                q = seed_mean(run)
                s = total(q, y)
                d, lo, hi = boot(s0, s, blocks, rng)
                n_in, _, n_ep, sec = run_stats(run)
                rows.append({'组': gname, '方案': name, '模型输入列': n_in, 'CRPS': score(q, y).mean(), 'twCRPS': tw(q, y).mean(), '总分': s.mean(),
                             '相对最终模型': f'{d:+.2%} [{lo:+.2%}, {hi:+.2%}]', '判断': verdict(lo, hi), '每轮秒数': sec / max(n_ep, 1), '总分钟': train_minutes(run)})
                xs.append(k); ys.append(s.mean())
            if xs:
                ax.plot(xs, ys, color=color, lw=1.5, marker=marker, ms=5, mec='white', mew=.6, label=gname.replace('13 个变量', '13 variables').replace('11 个成员', '11 ensemble members'))
        ax.axhline(s0.mean(), color=BASE, lw=1, ls='--', label=f'Final 7 inputs, no PCA ({s0.mean():.4f})')
        ax.set_title(m.upper() if m == 'gru' else m.capitalize(), loc='left', fontsize=10, color=INK)
        ax.set_xlabel('Number of principal components'); ax.set_ylabel('Validation CRPS + twCRPS (m)')
        ax.set_xticks(range(1, 14)); ax.invert_xaxis()
        ax.grid(color='#e8e7e2', lw=.6); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        lines += [f'## {m}（最终 7 个输入：总分 {s0.mean():.4f}）', ''] + (table(pd.DataFrame(rows)) if rows else ['还没有结果。']) + ['']
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'pca_sweep.png'))
    plt.close(fig)
    return lines


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('what', choices=['lambda', 'ensemble', 'pareto', 'pca', 'sweep'])
    p.add_argument('--t', default='GKH_h15_lam1_transformer_pretrain')
    p.add_argument('--g', default='GKH_h15_lam1_gru_pretrain')
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == 'lambda':
        lines, _ = do_lambda()
        out = os.path.join(OUT, 'lambda_grid.md')
    elif a.what == 'sweep':
        lines = do_sweep()
        out = os.path.join(OUT, 'pca_sweep.md')
    elif a.what == 'pca':
        lines = do_pca()
        out = os.path.join(OUT, 'pca_comparison.md')
    elif a.what == 'pareto':
        lines = do_pareto()
        out = os.path.join(OUT, 'pareto_lambda.md')
    else:
        lines, _ = do_ensemble(a.t, a.g)
        out = os.path.join(OUT, f'ensemble_weight_{a.t}_{a.g}.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
