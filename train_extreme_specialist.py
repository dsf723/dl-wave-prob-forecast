"""极端天气专用模型：极端天气在附近时切换到专用模型；专用模型从通用模型出发，只用极端天气样本继续微调。

"极端天气在附近"只用起报时已经知道的信息判断（决策时刻前 24 小时内）：
  - 台风：热带气旋中心在站点 --radius-km（默认 1000 km）以内；
  - 寒潮：偏北风（风向与正北夹角 ≤ 60°）风速 ≥ 10 m/s，且 24 小时降温 ≥ 8 ℃（阈值同配置 cold_*）。
预见期内才生成或移近的台风，起报时不知道，无法切换——这是可预报性的上限，专用模型解决不了。

做法（每个随机种子）：读取通用模型的检查点 → 学习率乘 --lr-factor（默认 0.3）→ 只用训练集里"极端天气在附近"的样本微调，
按验证集里同类样本的 CRPS 早停（--patience，默认 5）→ 验证集中这类样本改用专用模型，其余仍用通用模型（路由）。
输入标准化沿用通用模型的参数，保证两个模型看到的输入完全一致。测试集已用过，这里只在验证集上评价。
输出：<通用模型 output_dir>/extreme_specialist/ 下的 <模型>_seed<种子>.pt、routed_<模型>_val.npz、report.md
用法：python train_extreme_specialist.py model_configs/GKH_tuned_transformer_pretrain.json --model transformer
      python train_extreme_specialist.py model_configs/GKH_tuned_gru_pretrain.json --model gru
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

from project_paths import resolve_config        # noqa: E402
from run_training import Tee                     # noqa: E402


def extreme_flags(sites, samples, cfg, radius_km):
    """True when a cyclone within radius_km or a cold surge occurred in the 24 h before decision time."""
    import numpy as np
    import pandas as pd
    info = {}
    for c, obs, _ in sites:
        o = obs.copy()
        o['dist'] = o['label_tc_distance_km'] if 'label_tc_distance_km' in o else np.nan
        speed = np.hypot(o['u10'], o['v10'])
        direction = (np.degrees(np.arctan2(-o['u10'], -o['v10'])) + 360) % 360
        north = np.minimum(direction, 360 - direction) <= cfg['cold_north_sector_deg']
        drop = o['t2m'] - o['t2m'].reindex(o.index - pd.Timedelta(hours=24)).to_numpy()
        o['cold'] = north & (speed >= cfg['cold_wind_ms']) & (drop <= -cfg['cold_drop_24h_k'])
        info[str(c['site_id'])] = o[['dist', 'cold']]
    flags, kinds = [], []
    for s in samples:
        m = s['metadata'][0]
        o = info[str(m['site_id'])]
        d = pd.Timestamp(m['decision_time'])
        w = o.loc[(o.index > d - pd.Timedelta(hours=24)) & (o.index <= d)]
        tc = bool((w['dist'] <= radius_km).any())
        cold = bool(w['cold'].any())
        flags.append(tc or cold)
        kinds.append('typhoon' if tc else ('cold_surge' if cold else 'background'))
    return np.array(flags), np.array(kinds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--model', default='transformer', choices=['tide', 'transformer', 'gru', 'cnn_bigru'])
    p.add_argument('--radius-km', type=float, default=1000)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr-factor', type=float, default=0.3)
    a = p.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from wavecast.data import Scaler, feature_names
    from wavecast.experiment import load_sites, build_all, make_model, tensors, predict, write_json
    from wavecast.metrics import score
    from wavecast.models import crps_loss, twcrps_loss

    config = resolve_config(a.config)
    cfg = json.load(open(config, encoding='utf-8'))
    root = cfg['output_dir']
    out = os.path.join(root, 'extreme_specialist')
    os.makedirs(out, exist_ok=True)
    sys.stdout = sys.stderr = Tee(sys.__stdout__, open(os.path.join(out, f'log_{a.model}.txt'), 'a', encoding='utf-8', buffering=1))
    selection = json.load(open(os.path.join(root, 'selection.json'), encoding='utf-8'))
    ablation = selection['best'][a.model]['ablation']
    print(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} 极端天气专用模型：{a.model}_{ablation}，通用模型 {root}', flush=True)

    sites, _ = load_sites(cfg)
    samples, _, _ = build_all(sites, cfg, feature_names(cfg, ablation))
    ext_tr, kind_tr = extreme_flags(sites, samples['train'], cfg, a.radius_km)
    ext_va, kind_va = extreme_flags(sites, samples['val'], cfg, a.radius_km)
    print(f'极端天气在附近的起报：训练 {ext_tr.sum()} / {len(ext_tr)}（台风 {(kind_tr == "typhoon").sum()}，寒潮 {(kind_tr == "cold_surge").sum()}），'
          f'验证 {ext_va.sum()} / {len(ext_va)}（台风 {(kind_va == "typhoon").sum()}，寒潮 {(kind_va == "cold_surge").sum()}）', flush=True)
    if ext_tr.sum() < 20 or ext_va.sum() < 5:
        raise SystemExit('极端天气样本太少，无法可靠微调；可加大 --radius-km')
    train_ext = [s for s, f in zip(samples['train'], ext_tr) if f]
    val_ext = [s for s, f in zip(samples['val'], ext_va) if f]
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device'] == 'auto' else cfg['device']
    tw = cfg.get('tw_loss')
    lens = [len(s['y']) for s in samples['val']]
    row_ext = np.repeat(ext_va, lens)
    y_val = np.concatenate([s['y'] for s in samples['val']])

    routed_all, general_all = [], []
    for seed in cfg['seeds']:
        state = torch.load(os.path.join(root, f'{a.model}_{ablation}_seed{seed}.pt'), map_location=device, weights_only=False)
        scaler = Scaler()
        scaler.mean, scaler.std = np.array(state['scaler']['mean']), np.array(state['scaler']['std'])
        torch.manual_seed(seed)
        model = make_model(a.model, state['n_features'], state['history'], state['horizon'], cfg).to(device)
        model.load_state_dict(state['model'])
        general_val = np.load(os.path.join(root, f'{a.model}_{ablation}_seed{seed}_val.npz'))['quantiles']
        tr_ds, va_ds = tensors(train_ext, scaler), tensors(val_ext, scaler)
        loader = DataLoader(tr_ds, batch_size=cfg['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(seed))
        vl = DataLoader(va_ds, batch_size=cfg['batch_size'])
        yv = va_ds.tensors[2].numpy().ravel()
        best = float(score(predict(model, vl, device), yv).mean())
        print(f'[种子 {seed}] 通用模型在验证集极端样本上的 CRPS {best:.4f}（微调起点）', flush=True)
        ckpt = os.path.join(out, f'{a.model}_seed{seed}.pt')
        torch.save(model.state_dict(), ckpt)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'] * a.lr_factor, weight_decay=1e-4)
        stale = 0
        for epoch in range(a.epochs):
            t0 = time.time(); model.train(); total = count = 0
            for pp, ff, yy in loader:
                pp, ff, yy = pp.to(device), ff.to(device), yy.to(device)
                opt.zero_grad()
                q = model(pp, ff)
                loss = crps_loss(q, yy, model.head.tau)
                if tw:
                    loss = loss + tw['weight'] * twcrps_loss(q, yy, model.head.tau, tw['threshold_m'])
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step(); total += loss.item() * len(pp); count += len(pp)
            v = float(score(predict(model, vl, device), yv).mean())
            flag = ' *best' if v < best - 1e-6 else ''
            print(f'  微调 epoch {epoch + 1}/{a.epochs}: train {total / count:.4f}, 验证极端样本 CRPS {v:.4f}{flag}  ({time.time() - t0:.0f} s)', flush=True)
            if v < best - 1e-6:
                best, stale = v, 0
                torch.save(model.state_dict(), ckpt)
            else:
                stale += 1
                if stale >= a.patience:
                    break
        model.load_state_dict(torch.load(ckpt, map_location=device))
        spec_val = predict(model, vl, device)
        routed = general_val.copy()
        routed[row_ext] = spec_val
        routed_all.append(routed); general_all.append(general_val)
    routed = np.mean(routed_all, axis=0); general = np.mean(general_all, axis=0)
    np.savez_compressed(os.path.join(out, f'routed_{a.model}_val.npz'), quantiles=routed)
    sites_row = np.concatenate([[m['site_id'] for m in s['metadata']] for s in samples['val']])
    lines = ['| 分组 | 样本 | 通用模型 CRPS | 切换后 CRPS | 变化 |', '|---|---:|---:|---:|---:|']
    groups = [('全部', np.ones_like(row_ext)), ('极端天气在附近（起报时已知）', row_ext), ('其余（背景）', ~row_ext)]
    groups += [(f'站点 {z} 极端', row_ext & (sites_row == z)) for z in 'GKH']
    for name, k in groups:
        if k.sum():
            g, r = score(general[k], y_val[k]).mean(), score(routed[k], y_val[k]).mean()
            lines.append(f'| {name} | {int(k.sum())} | {g:.4f} | {r:.4f} | {r / g - 1:+.1%} |')
    write_json(os.path.join(out, f'settings_{a.model}.json'), dict(model=a.model, ablation=ablation, radius_km=a.radius_km,
               lr_factor=a.lr_factor, epochs=a.epochs, patience=a.patience,
               n_train_extreme=int(ext_tr.sum()), n_val_extreme=int(ext_va.sum())))
    open(os.path.join(out, f'report_{a.model}.md'), 'w', encoding='utf-8').write(
        f'# 极端天气专用模型（{a.model}，验证集）\n\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'=== 完成；结果 {out}', flush=True)


if __name__ == '__main__':
    main()
