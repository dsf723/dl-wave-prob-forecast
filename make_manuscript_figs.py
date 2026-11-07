"""论文英文插图。只读已有结果文件，不训练、不挑选。

输出 results/manuscript_figs/fig1–fig6.png（300 dpi）：
  fig1 框架示意；fig2 逐预见日 CRPSS（2017–2019 回算测试集 / 2020–2024 业务）；fig3 覆盖率、区间宽度与集成的 PIT 直方图；
  fig4 预报实例（测试集 2019-01-16 起报：三站都有完整实测的冬季起报中选一次，只作示例）；fig5 台湾海峡业务检验逐预见日 CRPS；fig6 波浪能资源（浮标逐月、ERA5 逐年）。
用法：python make_manuscript_figs.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from wavecast.metrics import score, TAU  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'manuscript_figs')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.linewidth': 0.8, 'savefig.dpi': 300})
TEST = os.path.join(ROOT, 'server_upload', 'test_results_1028', 'model_runs')
OPER = os.path.join(ROOT, 'server_upload', 'operational_results_1028', 'results', 'operational', 'GKH_h15_transformer_pretrain_ens_oper')
SITE_NAME = {'G': 'Guam (G)', 'K': 'Marado (K)', 'H': 'Hawaii (H)'}
COL = {'ensemble': '#1f4e9c', 'emos_ln': '#d9822b', 'emos_tn': '#8c6d31', 'raw_ensemble': '#c0392b', 'climatology': '#7f7f7f'}
LAB = {'ensemble': 'Transformer–GRU ensemble', 'emos_ln': 'EMOS (log-normal)', 'emos_tn': 'EMOS (truncated normal)',
       'raw_ensemble': 'Raw GEFS ensemble', 'climatology': 'Climatology'}


def test_table():
    """测试集逐时刻：各基准 CRPS（case_scores），集成 = 两模型分位数平均后现算。"""
    keys = ['site_id', 'issue_time', 'valid_time']
    ct = pd.read_csv(os.path.join(TEST, 'GKH_h15_transformer_pretrain', 'test', 'case_scores.csv'),
                     usecols=keys + ['lead_day', 'model', 'truth', 'crps', 'covered80', 'width80', 'p10', 'p50', 'p90'])
    tr = ct[ct.model == 'transformer'].reset_index(drop=True)
    qa = np.load(os.path.join(TEST, 'GKH_h15_transformer_pretrain', 'test', 'quantiles.npz'))['transformer']
    cg = pd.read_csv(os.path.join(TEST, 'GKH_h15_gru_pretrain', 'test', 'case_scores.csv'), usecols=keys + ['model'])
    gr = cg[cg.model == 'gru'].reset_index(drop=True)
    qb = np.load(os.path.join(TEST, 'GKH_h15_gru_pretrain', 'test', 'quantiles.npz'))['gru']
    assert (tr[keys].values == gr[keys].values).all() and len(tr) == len(qa) == len(qb)
    q = (qa + qb) / 2
    y = tr.truth.to_numpy(float)
    ens = tr[keys + ['lead_day', 'truth']].copy()
    ens['model'] = 'ensemble'; ens['crps'] = score(q, y)
    ens['covered80'] = (y >= q[:, 9]) & (y <= q[:, 89]); ens['width80'] = q[:, 89] - q[:, 9]
    ens['p10'], ens['p50'], ens['p90'] = q[:, 9], q[:, 49], q[:, 89]
    ens['pit'] = [np.interp(v, qq, TAU, left=0.0, right=1.0) for v, qq in zip(y, q)]
    base = ct[ct.model.isin(['emos_ln', 'emos_tn', 'raw_ensemble', 'climatology'])]
    return pd.concat([ens, base], ignore_index=True)


def oper_table():
    c = pd.read_csv(os.path.join(OPER, 'case_scores.csv'), usecols=['lead_day', 'model', 'crps', 'covered80', 'width80'])
    return c[c.model.isin(['ensemble', 'emos_ln', 'emos_tn', 'raw_ensemble', 'climatology'])]


def crpss(t):
    m = t.groupby(['model', 'lead_day']).crps.mean().unstack(0)
    return 1 - m.div(m['climatology'], axis=0)


def fig1():
    fig, ax = plt.subplots(figsize=(7.6, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.5', fc=fc, ec='#333333', lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.6, linespacing=1.25)
    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.9, color='#333333'))
    box(1, 30, 21, 13, 'GEFSv12 wave ensemble\n11 members, days 0–15,\n3-hourly: mean/spread of $H_s$,\n$T_e$, direction, season', '#e8eef8')
    box(1, 12, 21, 13, 'Past 7 days\nbuoy $H_s$ + ERA5 $T_e$,\ndirection;\nsite one-hot code', '#e8eef8')
    box(1, 0.5, 21, 7.5, 'Pre-training: ERA5 as\nproxy truth (20 epochs)', '#f4f4f4')
    box(29, 26, 17, 13, 'Transformer\nencoder\n(3 seeds)', '#fdf1e3')
    box(29, 7, 17, 13, 'GRU\nencoder\n(3 seeds)', '#fdf1e3')
    box(52, 16, 18, 14, 'Bernstein quantile\nhead (99 quantiles)\nloss: CRPS +\ntwCRPS (2.5 m)', '#fdf1e3')
    box(76, 26, 22, 13, 'Vincentized ensemble\n(quantile averaging)\ndays 1–15', '#e6f2e6')
    box(76, 7, 22, 13, 'Product: days 14–15 blend,\ndays 16–30 climatology;\n$T_e$ head, wave power', '#e6f2e6')
    arrow(22.8, 36.5, 28.6, 33); arrow(22.8, 18.5, 28.6, 13.5); arrow(22.8, 36, 28.6, 14.5); arrow(22.8, 19, 28.6, 31.5)
    arrow(46.6, 32, 51.6, 25); arrow(46.6, 13.5, 51.6, 21); arrow(70.6, 25, 75.6, 31); arrow(87, 25.6, 87, 20.6)
    arrow(11.5, 8.6, 11.5, 11.4)
    fig.savefig(os.path.join(OUT, 'fig1_framework.png'), bbox_inches='tight'); plt.close(fig)


def fig2(tt, to):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, t, title in [(axs[0], tt, '(a) Reforecast test set, 2017–2019'), (axs[1], to, '(b) Operational GEFS, 2020–2024')]:
        s = crpss(t)
        for m in ['ensemble', 'emos_ln', 'emos_tn', 'raw_ensemble']:
            ax.plot(s.index, s[m], marker='o', ms=3, lw=1.3, color=COL[m], label=LAB[m])
        ax.axhline(0, color='k', lw=0.7); ax.set_xlabel('Lead time (days)'); ax.set_title(title, fontsize=10)
        ax.set_xticks(range(1, 16, 2)); ax.grid(alpha=0.3, lw=0.5)
    axs[0].set_ylabel('CRPSS relative to climatology'); axs[0].legend(fontsize=8, frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_crpss_lead.png')); plt.close(fig)


def fig3(tt):
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
    g = tt.groupby(['model', 'lead_day'])
    cov, wid = g.covered80.mean().unstack(0) * 100, g.width80.mean().unstack(0)
    for m in ['ensemble', 'emos_ln', 'raw_ensemble', 'climatology']:
        axs[0].plot(cov.index, cov[m], marker='o', ms=3, lw=1.3, color=COL[m], label=LAB[m])
        axs[1].plot(wid.index, wid[m], marker='o', ms=3, lw=1.3, color=COL[m], label=LAB[m])
    axs[0].axhline(80, color='k', ls='--', lw=0.8); axs[0].set_ylabel('P10–P90 coverage (%)'); axs[0].set_title('(a) Calibration', fontsize=10)
    axs[1].set_ylabel('P10–P90 width (m)'); axs[1].set_title('(b) Sharpness', fontsize=10)
    for ax in axs[:2]:
        ax.set_xlabel('Lead time (days)'); ax.set_xticks(range(1, 16, 2)); ax.grid(alpha=0.3, lw=0.5)
    axs[1].legend(fontsize=7, frameon=False, loc='upper left')
    pit = tt.loc[tt.model == 'ensemble', 'pit'].to_numpy(float)
    axs[2].hist(pit, bins=20, range=(0, 1), density=True, color=COL['ensemble'], alpha=0.8, edgecolor='white')
    axs[2].axhline(1, color='k', ls='--', lw=0.8); axs[2].set_xlabel('PIT'); axs[2].set_ylabel('Density')
    axs[2].set_title('(c) PIT, ensemble', fontsize=10); axs[2].set_ylim(0, 1.6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_coverage_width.png')); plt.close(fig)


def fig4(tt):
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
    issue = '2019-01-16 00:00:00+00:00'
    for ax, s in zip(axs, ['G', 'K', 'H']):
        e = tt[(tt.model == 'ensemble') & (tt.site_id == s) & (tt.issue_time == issue)].copy()
        r = tt[(tt.model == 'raw_ensemble') & (tt.site_id == s) & (tt.issue_time == issue)].copy()
        if e.empty:
            ax.text(0.5, 0.5, f'{SITE_NAME[s]}: no test forecast for this issue', transform=ax.transAxes, ha='center'); continue
        for d in (e, r):
            d['t'] = pd.to_datetime(d.valid_time, utc=True, format='mixed')
        e, r = e.sort_values('t'), r.sort_values('t')
        ax.fill_between(e.t, e.p10, e.p90, color=COL['ensemble'], alpha=0.25, lw=0, label='Ensemble P10–P90')
        ax.plot(e.t, e.p50, color=COL['ensemble'], lw=1.4, label='Ensemble P50')
        ax.plot(r.t, r.p10, color=COL['raw_ensemble'], lw=0.8, ls='--', label='Raw GEFS P10 / P90')
        ax.plot(r.t, r.p90, color=COL['raw_ensemble'], lw=0.8, ls='--')
        ax.plot(e.t, e.truth, 'k.', ms=2.5, label='Buoy observation')
        ax.set_ylabel('$H_s$ (m)'); ax.set_title(f'({"abc"[["G", "K", "H"].index(s)]}) {SITE_NAME[s]}', fontsize=10, loc='left'); ax.grid(alpha=0.3, lw=0.5)
    axs[0].legend(fontsize=7.5, frameon=False, ncol=4, loc='upper right')
    axs[-1].set_xlabel('Valid time (UTC), forecast issued 2019-01-16 00 UTC')
    import matplotlib.dates as mdates
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_example_forecast.png')); plt.close(fig)


def fig5():
    d = os.path.join(ROOT, 'results', 'operational', 'TWS_h15_transformer_pretrain_ens_oper_TW46757B_TWC6W08')
    c = pd.read_csv(os.path.join(d, 'cases.csv')); z = np.load(os.path.join(d, 'quantiles.npz')); y = c.truth.to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for m in ['ensemble', 'emos_tn', 'raw_ensemble', 'climatology']:
        s = pd.Series(score(z[m], y)).groupby(c.lead_day.values).mean()
        ax.plot(s.index, s.values, marker='o', ms=3, lw=1.3, color=COL[m], label=LAB[m])
    ax.set_xlabel('Lead time (days)'); ax.set_ylabel('CRPS (m)'); ax.set_xticks(range(1, 16, 2)); ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=8, frameon=False); ax.set_title('Taiwan Strait (Matsu, Hsinchu), operational GEFS 2020–2024', fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig5_taiwan_oper_crps.png')); plt.close(fig)


def fig6():
    R = os.path.join(ROOT, 'processed')
    def rd(p, cols):
        d = pd.read_csv(p, usecols=['valid_time'] + cols)
        d.index = pd.to_datetime(d.valid_time.str.replace('Z', '')).dt.floor('h')
        return d.drop(columns='valid_time').groupby(level=0).mean()
    series = {}
    for k, spec in [('Guam (G)', 'G_121p1'), ('Hawaii (H)', 'H_198p1')]:
        s = rd(os.path.join(R, 'buoy_spectral', f'{spec}.csv'), ['hs_spec', 'te']).rename(columns={'hs_spec': 'hs'}); series[k] = s
    kb = rd(os.path.join(R, 'buoy', 'K_22107.csv'), ['hs']); ke = rd(os.path.join(R, 'site_K', 'observations_hourly.csv'), ['te'])
    series['Marado (K)'] = kb.join(ke, how='inner')
    hb = rd(os.path.join(R, 'buoy', 'CW46757B_46757B.csv'), ['hs', 'tmean_cwa']); hb['te'] = hb.tmean_cwa / 0.863; series['Hsinchu'] = hb
    mb = rd(os.path.join(R, 'buoy', 'CWC6W08_C6W08.csv'), ['hs', 'tp']); mb['te'] = mb.tp * 0.9; series['Matsu'] = mb
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9))
    cols = ['#1f4e9c', '#c0392b', '#2e8b57', '#d9822b', '#7d3c98']
    order = ['Guam (G)', 'Hawaii (H)', 'Marado (K)', 'Hsinchu', 'Matsu']
    for k, cc in zip(order, cols):
        d = series[k]
        e = d[(d.hs > 0) & (d.hs < 25)].dropna(subset=['hs', 'te'])
        p = (0.4906 * e.hs ** 2 * e.te).groupby(e.index.month).mean()
        axs[0].plot(p.index, p.values, marker='o', ms=3, lw=1.3, color=cc, label=k)
    axs[0].set_xticks(range(1, 13)); axs[0].set_xlabel('Month'); axs[0].set_ylabel('Mean wave power (kW m$^{-1}$)')
    axs[0].set_title('(a) Monthly mean, buoy observations', fontsize=10); axs[0].legend(fontsize=7.5, frameon=False); axs[0].grid(alpha=0.3, lw=0.5)
    y = pd.read_csv(os.path.join(ROOT, 'results', 'resource_metrics', 'yearly_metrics.csv'), encoding='utf-8-sig')
    sc = {'G': ('Guam (G)', cols[0]), 'H': ('Hawaii (H)', cols[1]), 'K': ('Marado (K)', cols[2])}
    for s_, (lab, cc) in sc.items():
        g = y[y.site == s_]
        axs[1].plot(g.year, g.mean_power, lw=1.2, color=cc, label=lab)
        axs[2].plot(g.year, g.mvi, lw=1.2, color=cc, label=lab)
    axs[1].set_ylabel('Annual mean wave power (kW m$^{-1}$)'); axs[1].set_title('(b) Annual mean, ERA5', fontsize=10)
    axs[2].set_ylabel('MVI'); axs[2].set_title('(c) MVI, ERA5', fontsize=10)
    for ax in axs[1:]:
        ax.set_xlabel('Year'); ax.grid(alpha=0.3, lw=0.5)
    axs[1].legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig6_wave_power.png')); plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    tt, to = test_table(), oper_table()
    fig1(); fig2(tt, to); fig3(tt); fig4(tt); fig5(); fig6()
    s, so = crpss(tt), crpss(to)
    print('test CRPSS ensemble by lead:', s['ensemble'].round(3).tolist())
    print('oper CRPSS ensemble by lead:', so['ensemble'].round(3).tolist())
    print('test overall CRPS:', tt.groupby('model').crps.mean().round(4).to_dict())
    print('→', OUT)


if __name__ == '__main__':
    main()
