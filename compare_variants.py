"""改进方案在验证集上的对比（测试集已用过，改进只在验证集上比较，作为补充分析）。

比较对象（都在同一批验证样本上、按站点拆分）：
  原始集合、调参后 Transformer（最终模型）、调参后 GRU、二者集成（分位数平均）、
  不带站点编号的三站联合、留一站（该站不参加训练，用另外两站训练后预报它）。
指标：CRPS、阈值加权 CRPS（2.5 m）、80% 覆盖率、有效时刻实况台风期 CRPS、起报时已知台风（1000 km）CRPS。
用法：python compare_variants.py        输出 results/improvements/validation_comparison.csv 与屏幕表格
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from project_paths import resolve_config                          # noqa: E402
from wavecast.data import feature_names                            # noqa: E402
from wavecast.experiment import load_sites, build_all              # noqa: E402
from wavecast.metrics import score, quantile_at                    # noqa: E402
from wavecast.tc_gate import decision_tc_distance                  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'improvements')
REF = 'model_configs/GKH_reforecast_joint_tw_tuned_transformer.json'


def val_quantiles(run, model, ablation='season'):
    cfg = json.load(open(resolve_config(f'model_configs/{run}.json'), encoding='utf-8'))
    root = cfg['output_dir']
    files = [os.path.join(root, f'{model}_{ablation}_seed{s}_val.npz') for s in cfg['seeds']]
    if not all(os.path.exists(f) for f in files):
        return None
    return np.mean([np.load(f)['quantiles'] for f in files], axis=0)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(resolve_config(REF), encoding='utf-8'))
    sites, _ = load_sites(cfg)
    samples, _, _ = build_all(sites, cfg, feature_names(cfg, 'season'))
    val = samples['val']
    y = np.concatenate([s['y'] for s in val])
    site = np.concatenate([[m['site_id'] for m in s['metadata']] for s in val])
    weather = np.concatenate([[m['weather'] for m in s['metadata']] for s in val])
    dist = decision_tc_distance(sites, val)
    tc_near = np.repeat(dist <= 1000, [len(s['y']) for s in val])
    raw = np.concatenate([s['raw_quantiles'] for s in val])
    from train_extreme_specialist import extreme_flags
    ext_row = np.repeat(extreme_flags(sites, val, cfg, 1000)[0], [len(s['y']) for s in val])

    variants = {'原始集合': raw}
    tr = val_quantiles('GKH_reforecast_joint_tw_tuned_transformer', 'transformer')
    gru = val_quantiles('GKH_reforecast_joint_tw_tuned_gru', 'gru')
    variants['Transformer（最终模型）'] = tr
    if gru is not None:
        variants['GRU（调参后）'] = gru
        variants['集成 Transformer+GRU'] = (tr + gru) / 2
    noh = val_quantiles('GKH_tuned_transformer_noonehot', 'transformer')
    if noh is not None:
        variants['三站联合·无站点编号'] = noh
    trp = val_quantiles('GKH_tuned_transformer_pretrain', 'transformer')
    grp = val_quantiles('GKH_tuned_gru_pretrain', 'gru')
    if trp is not None:
        variants['Transformer + ERA5 预训练'] = trp
    if grp is not None:
        variants['GRU + ERA5 预训练'] = grp
    if trp is not None and grp is not None:
        variants['集成（均带预训练）'] = (trp + grp) / 2
    def routed(run, model):
        cfg_r = json.load(open(resolve_config(f'model_configs/{run}.json'), encoding='utf-8'))
        f = os.path.join(cfg_r['output_dir'], 'extreme_specialist', f'routed_{model}_val.npz')
        return np.load(f)['quantiles'] if os.path.exists(f) else None
    rtr, rgr = routed('GKH_tuned_transformer_pretrain', 'transformer'), routed('GKH_tuned_gru_pretrain', 'gru')
    if rtr is not None:
        variants['Transformer + 预训练 + 极端专用'] = rtr
    if rgr is not None:
        variants['GRU + 预训练 + 极端专用'] = rgr
    if rtr is not None and rgr is not None:
        variants['集成 + 预训练 + 极端专用'] = (rtr + rgr) / 2
    loso = {h: val_quantiles(f'LOSO_holdout_{h}', 'transformer') for h in 'GKH'}
    if all(v is not None for v in loso.values()):
        q = np.empty_like(tr)
        for h, v in loso.items():
            q[site == h] = v[site == h]          # each site predicted by the model that never saw it
        variants['留一站（该站未参加训练）'] = q

    rows = []
    for name, q in variants.items():
        if q is None or len(q) != len(y):
            print(f'[跳过] {name}：没有结果或样本数不一致')
            continue
        crps = score(q, y)
        tw = score(np.maximum(q, 2.5), np.maximum(y, 2.5))
        cov = (y >= quantile_at(q, .1)) & (y <= quantile_at(q, .9))
        r = dict(方案=name, 三站CRPS=crps.mean(), 覆盖率=cov.mean())
        for s in 'GKH':
            r[f'{s}_CRPS'] = crps[site == s].mean()
            r[f'{s}_twCRPS'] = tw[site == s].mean()
        r['K实况台风期'] = crps[(site == 'K') & (weather == 'typhoon')].mean()
        r['起报时极端天气'] = crps[ext_row].mean()
        r['起报时已知台风'] = crps[tc_near].mean()
        rows.append(r)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(OUT, 'validation_comparison.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print(res.round(4).to_string(index=False))
    print(f'\n验证集，CRPS 单位 m，越小越好；结果 {OUT}')


if __name__ == '__main__':
    main()
