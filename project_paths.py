"""配置文件路径的可移植处理：让同一份配置在本机（F:\\Offshore Power Forecasting）和云服务器上都能用。

配置里的 data_dir、output_dir、sites[*].data_dir、pretrain.sites[*].data_dir：
  - 相对路径 → 以项目根目录（本文件所在目录）为基准；
  - 写的是本机绝对路径（F:/Offshore Power Forecasting/...）但在当前机器上不存在 → 换成当前项目根目录。
换过路径的配置另存为 model_configs/_resolved/<原文件名>，返回新路径；没有变化则返回原路径。
注意：配置的哈希值包含路径，所以在哪台机器上做验证，就在哪台机器上做测试。
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_ROOTS = ['F:/Offshore Power Forecasting', 'F:\\Offshore Power Forecasting']


def _fix(path):
    if not path:
        return path
    if not os.path.isabs(path) and not path.startswith(('F:', 'f:')):
        return os.path.join(ROOT, path).replace('\\', '/')
    if os.path.exists(path) or os.path.exists(os.path.dirname(path.rstrip('/\\')) or path):
        return path
    norm = path.replace('\\', '/')
    for prefix in LOCAL_ROOTS:
        p = prefix.replace('\\', '/')
        if norm.lower().startswith(p.lower()):
            return (ROOT.replace('\\', '/') + norm[len(p):])
    return path


def resolve_config(config_path):
    config_path = os.path.abspath(config_path)
    cfg = json.load(open(config_path, encoding='utf-8'))
    new = json.loads(json.dumps(cfg))
    for key in ('data_dir', 'output_dir'):
        if key in new:
            new[key] = _fix(new[key])
    for site in new.get('sites', []) + (new.get('pretrain') or {}).get('sites', []):   # 10-22：预训练站点也要换
        site['data_dir'] = _fix(site['data_dir'])
    if new == cfg:
        return config_path
    out_dir = os.path.join(ROOT, 'model_configs', '_resolved')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(config_path))
    json.dump(new, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'    路径已换成本机项目目录：{out}', flush=True)
    return out
