"""EMOS 基准、CRPS 可靠度 / 分辨率分解、大浪条件 PIT 图、共形覆盖率校正（验证集）。

1. EMOS 基准（Gneiting 2005）：截断正态（TN）与对数正态（LN）两种，每站 × 预见天各一组系数，
   输入 = GEFS 集合平均 Hs 与集合离散度，训练年最小 CRPS 拟合；与本文模型、GEFS 原始集合、气候态比较。
2. Hersbach（2000）分解：CRPS = 可靠度 + 分辨率之外的潜在 CRPS；分辨率 = 不确定度 − 潜在 CRPS（越大越好，可靠度越小越好）。
3. PIT 直方图：全部样本；只看大浪（实测 > 2.5 m、实测 > 本站训练年 90% 分位）时用条件 PIT（Allen 2023），
   即预报分布在阈值以上部分的 PIT，校准良好时应均匀。
4. 共形分位数回归（Romano 2019，单侧 / 非对称形式，逐分位数）：按 站点 × 预见期段 求每个分位数的平移量。
   验证集内按年份留一交叉验证评估（用其他年份定平移量，评价留出的年份）；最终平移量用全部验证年份，
   存为 JSON，供测试集与导出使用（测试集只用一次）。
用法：python postproc_diagnostics.py --run GKH_h15_transformer_pretrain [--b GKH_h15_gru_pretrain] [--split test]
（10-27 起最终方案为集成：加 --b；--split test 为测试集最终评价，须先在验证集上跑一次以生成共形平移量）
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from project_paths import resolve_config                                     # noqa: E402
from route_models import seed_mean, boot                                     # noqa: E402
from wavecast.data import feature_names                                      # noqa: E402
from wavecast.experiment import load_sites, build_all, climatology           # noqa: E402
from wavecast.metrics import TAU, score                                      # noqa: E402
from wavecast.postproc import (emos_benchmark, hersbach, pit, conditional_pit,  # noqa: E402
                               conformal_shifts, apply_shifts)

OUT = os.path.join(ROOT, 'results', 'postproc')


def tw(q, y, t):
    return score(np.maximum(q, t), np.maximum(y, t))


def summary_row(name, q, y, t):
    return dict(system=name, n=len(y), crps=score(q, y).mean(), twcrps=tw(q, y, t).mean(),
                cover80=((y >= q[:, 9]) & (y <= q[:, 89])).mean(), below_p10=(y < q[:, 9]).mean(),
                above_p90=(y > q[:, 89]).mean(), width80=(q[:, 89] - q[:, 9]).mean())


def md(df, fmt='{:.4f}'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + ' |')
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='GKH_h15_transformer_pretrain')
    p.add_argument('--b', help='可选：第二个模型（如 GRU），加入两者分位数平均的集成')
    p.add_argument('--threshold', type=float, default=2.5)
    p.add_argument('--split', choices=['val', 'test'], default='val',
                   help='test：最终评价（测试集只用一次），共形平移量用验证集定好的 JSON，不重新拟合')
    a = p.parse_args()
    run_dir = os.path.join(ROOT, 'model_runs', a.run)
    cfg = json.load(open(resolve_config(os.path.join(ROOT, 'model_configs', f'{a.run}.json')), encoding='utf-8'))
    best = json.load(open(os.path.join(run_dir, 'selection.json'), encoding='utf-8'))['best']
    model, info = next(iter(best.items()))
    sites, _ = load_sites(cfg)
    samples, _, names = build_all(sites, cfg, feature_names(cfg, info['ablation']))
    train, val = samples['train'], samples[a.split]
    meta = pd.DataFrame([m for s in val for m in s['metadata']])
    meta['site_id'] = meta.site_id.astype(str)
    y = np.concatenate([s['y'] for s in val]).astype(float)
    t = a.threshold
    def quantiles(run):
        if a.split == 'val':
            return seed_mean(run)
        m = next(iter(json.load(open(os.path.join(ROOT, 'model_runs', run, 'selection.json'), encoding='utf-8'))['best']))
        path = os.path.join(ROOT, 'model_runs', run, 'test', 'quantiles.npz')
        if not os.path.exists(path):
            raise SystemExit(f'{run} 还没有测试集结果：先 python run_training.py model_configs/{run}.json test')
        return np.load(path)[m]
    systems = {'本文模型': quantiles(a.run)}
    if len(systems['本文模型']) != len(y):
        raise SystemExit('模型预报行数与样本不一致')
    if a.b:
        systems['集成（A、B 分位数平均）'] = (systems['本文模型'] + quantiles(a.b)) / 2
    # the final scheme (10-27: Transformer + GRU ensemble when --b is given) is the one compared with EMOS and calibrated
    final = '集成（A、B 分位数平均）' if a.b else '本文模型'
    tag = f'{a.run}_ens' if a.b else a.run
    val_tag = tag
    if a.split == 'test':
        tag += '_test'
    print('拟合 EMOS（每站 × 预见天）…', flush=True)
    emos = emos_benchmark(train, val, names)
    systems['EMOS 截断正态'] = emos['tn']; systems['EMOS 对数正态'] = emos['ln']
    systems['GEFS 原始集合'] = np.concatenate([s['raw_quantiles'] for s in val]).astype(float)
    systems['气候态'] = climatology(sites, val, cfg['target'])
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f'{a.run}_{a.split}_emos.npz'), tn=emos['tn'], ln=emos['ln'])
    rep = [f'# 后处理诊断：{tag}（{"验证集" if a.split == "val" else "测试集"} {len(y)} 个时刻）', '']
    blocks = meta.block_id.to_numpy()
    rng = np.random.default_rng(0)

    # 1. EMOS benchmark
    rows = pd.DataFrame([summary_row(k, q, y, t) for k, q in systems.items()])
    rep += ['## 1. 与 EMOS 基准比较', '', f'twCRPS 阈值 {t} m；覆盖率为 P10–P90（理想 80%）。', '', md(rows), '']
    # on the test set the EMOS variant is the one that was better on validation (log-normal, 10-26), not re-chosen
    best_emos = (min(('EMOS 截断正态', 'EMOS 对数正态'), key=lambda k: score(systems[k], y).mean())
                 if a.split == 'val' else 'EMOS 对数正态')
    sm, se = score(systems[final], y), score(systems[best_emos], y)
    rep += [f'{final}相对 {best_emos}（按 block_id 配对自助 2000 次）：', '',
            f'| 分组 | {final} CRPS | EMOS CRPS | 变化 | 95% 区间 |', '|---|---:|---:|---:|---|']
    groups = [('全部', np.ones(len(y), bool))] + [(f'站点 {s}', (meta.site_id == s).to_numpy()) for s in meta.site_id.unique()]
    groups += [(b, (meta.lead_band == b).to_numpy()) for b in meta.lead_band.unique()]
    groups += [('事后实况台风 / 寒潮', (meta.typhoon | meta.cold_surge).to_numpy().astype(bool))]
    for gname, k in groups:
        d, lo, hi = boot(se[k], sm[k], blocks[k], rng)
        rep.append(f'| {gname} | {sm[k].mean():.4f} | {se[k].mean():.4f} | {d:+.1%} | [{lo:+.1%}, {hi:+.1%}]'
                   f'{" 显著" if hi < 0 or lo > 0 else " 不显著"} |')
    rep.append('')

    # 2. Hersbach decomposition
    recs = []
    for gname, k in [('全部', np.ones(len(y), bool))] + [(f'{s} {b}', ((meta.site_id == s) & (meta.lead_band == b)).to_numpy())
                                                         for s in meta.site_id.unique() for b in meta.lead_band.unique()]:
        for name, q in systems.items():
            d = hersbach(q[k], y[k])
            recs.append(dict(分组=gname, 系统=name, CRPS=d['crps'], 可靠度=d['reliability'], 分辨率=d['resolution'],
                             不确定度=d['uncertainty']))
    dec = pd.DataFrame(recs)
    dec.to_csv(os.path.join(OUT, f'{tag}_hersbach.csv'), index=False, encoding='utf-8-sig')
    rep += ['## 2. CRPS 分解（Hersbach 2000）', '',
            'CRPS = 可靠度 − 分辨率 + 不确定度。可靠度越小越好（0 = 完全校准），分辨率越大越好；不确定度只取决于实测，同一分组各系统相同。'
            '全部分组见 CSV。', '', md(dec[dec.分组 == '全部'].drop(columns='分组')), '']

    # 3. PIT and conditional PIT
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    q90 = {str(c['site_id']): obs.loc[obs.index.year.isin(c['train_years']), cfg['target']].quantile(.9) for c, obs, _ in sites}
    site_t = meta.site_id.map(q90).to_numpy(float)
    show = [k for k in systems if k != '气候态']
    panels = [('全部样本（PIT）', None), (f'实测 > {t} m（条件 PIT）', np.full(len(y), t)),
              ('实测 > 本站训练年 90% 分位（条件 PIT）', site_t)]
    fig, axes = plt.subplots(len(panels), len(show), figsize=(3.2 * len(show), 2.6 * len(panels)), sharey='row')
    pit_rows = []
    for i, (title, thr) in enumerate(panels):
        for j, name in enumerate(show):
            q = systems[name]
            if thr is None:
                v, n = pit(q, y, rng), len(y)
            else:
                vs, ks = [], []
                for s in np.unique(thr):
                    m = thr == s
                    c, kk = conditional_pit(q[m], y[m], s, rng)
                    vs.append(c); ks.append(kk.sum())
                v, n = np.concatenate(vs), int(np.sum(ks))
            h = np.histogram(v, bins=10, range=(0, 1))[0] / max(len(v), 1)
            ax = axes[i, j]
            ax.bar(np.arange(10) / 10 + .05, h, width=.09, color='#4C72B0')
            ax.axhline(.1, color='k', lw=.8, ls='--')
            ax.set_xlim(0, 1)
            if i == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(title, fontsize=8)
            ax.text(.02, .92, f'n={n}', transform=ax.transAxes, fontsize=7)
            pit_rows.append(dict(样本=title, 系统=name, n=n, 最低一成=h[0], 最高一成=h[-1], 平均PIT=float(np.mean(v)) if len(v) else np.nan,
                                 偏离均匀最大=float(np.abs(h - .1).max())))
    fig.tight_layout()
    fig_path = os.path.join(OUT, f'{tag}_pit.png')
    fig.savefig(fig_path, dpi=150); plt.close(fig)
    rep += ['## 3. PIT 与大浪条件 PIT（Allen 2023）', '',
            f'图：`{os.path.relpath(fig_path, ROOT)}`。理想为每格 0.1；"最高一成"偏大 = 实测常高于预报的上尾（大浪报小），'
            '"最低一成"偏大 = 预报偏大。条件 PIT 只在实测超过阈值的时刻、用预报分布在阈值以上的部分计算。', '',
            md(pd.DataFrame(pit_rows), '{:.3f}'), '']

    # 4. conformal correction, leave-one-year-out within validation
    grp = (meta.site_id + '|' + meta.lead_band).to_numpy()
    year = pd.to_datetime(meta.issue_time, utc=True).dt.year.to_numpy()
    qm = systems[final]
    qc = np.array(qm, float)
    spath = os.path.join(OUT, f'{val_tag}_conformal_shifts.json')
    if a.split == 'test':
        if not os.path.exists(spath):
            raise SystemExit(f'缺少验证集定的平移量 {spath}：先在验证集上跑本脚本（同样的 --run / --b）')
        qc = apply_shifts(qm, grp, {g: np.asarray(v) for g, v in json.load(open(spath, encoding='utf-8'))['shifts'].items()})
    else:
        for yr in np.unique(year):
            k = year == yr
            sh = conformal_shifts(qm[~k], y[~k], grp[~k])
            qc[k] = apply_shifts(qm[k], grp[k], sh)
    cmp = []
    for gname, k in [('全部', np.ones(len(y), bool))] + [(b, (meta.lead_band == b).to_numpy()) for b in meta.lead_band.unique()] \
            + [(f'站点 {s}', (meta.site_id == s).to_numpy()) for s in meta.site_id.unique()]:
        r0, r1 = summary_row('校正前', qm[k], y[k], t), summary_row('共形校正后', qc[k], y[k], t)
        d, lo, hi = boot(score(qm[k], y[k]), score(qc[k], y[k]), blocks[k], rng)
        cmp.append(dict(分组=gname, 覆盖率_前=r0['cover80'], 覆盖率_后=r1['cover80'], 低于P10_后=r1['below_p10'],
                        高于P90_后=r1['above_p90'], CRPS_前=r0['crps'], CRPS_后=r1['crps'],
                        CRPS变化=f'{d:+.2%} [{lo:+.2%}, {hi:+.2%}]{" 显著" if hi < 0 or lo > 0 else ""}',
                        twCRPS_前=r0['twcrps'], twCRPS_后=r1['twcrps']))
    how = ('验证集内按年份留一交叉验证' if a.split == 'val' else '测试集，平移量为验证集定好的，未重新拟合')
    rep += [f'## 4. 共形分位数回归校正（{final}；Romano 2019；{how}）', '',
            '每个 站点 × 预见期段 分别求 99 个分位数的平移量（单侧 CQR：该分位数的残差 y − q 的 ⌈(n+1)τ⌉/n 经验分位数），'
            '平移后重新排序保证分位数单调。', '', md(pd.DataFrame(cmp), '{:.3f}'), '']
    if a.split == 'test':
        path = os.path.join(OUT, f'{tag}_report.md')
        open(path, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
        print('\n'.join(rep)); print(f'\n报告 → {path}')
        return
    shifts = conformal_shifts(qm, y, grp)
    json.dump(dict(run=a.run, ensemble_with=a.b, groups='site_id|lead_band', horizon_days=cfg['horizon_days'],
                   note='fitted on all validation years; apply with wavecast.postproc.apply_shifts',
                   shifts={g: v.round(4).tolist() for g, v in shifts.items()}),
              open(spath, 'w', encoding='utf-8'), ensure_ascii=False)
    rep += [f'最终平移量（全部验证年份）：`{os.path.relpath(spath, ROOT)}`。P10 / P90 平移（m）：', '',
            '| 组 | P10 平移 | P50 平移 | P90 平移 |', '|---|---:|---:|---:|']
    rep += [f'| {g} | {v[9]:+.3f} | {v[49]:+.3f} | {v[89]:+.3f} |' for g, v in sorted(shifts.items())]
    path = os.path.join(OUT, f'{tag}_report.md')
    open(path, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    print(f'\n报告 → {path}')


if __name__ == '__main__':
    main()
