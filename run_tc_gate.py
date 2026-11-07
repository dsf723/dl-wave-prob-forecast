"""台风期门控启动器：屏幕显示 + 写日志（<output_dir>/tc_gate/gate_log_<阶段>.txt）。

不重新训练：读取已完成试验的验证集预测，按"起报时台风是否在 1000 km 内 × 预见期段"选择模型与原始集合的混合权重。
用法（在项目根目录）：
  python run_tc_gate.py model_configs/GKH_reforecast_joint_tw.json validate
  python run_tc_gate.py model_configs/GKH_reforecast_joint_tw.json validate --radius-km 800 --min-events 5
  python run_tc_gate.py model_configs/GKH_reforecast_joint_tw.json test     （须先跑该试验的测试阶段；只跑一次）
其余参数：--model tide|transformer（默认 transformer）、--threshold-m 2.5、--tw-weight 1.0
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, 'outputs', 'wave_forecast')
sys.path.insert(0, ROOT)
sys.path.insert(0, PKG)

from project_paths import resolve_config          # noqa: E402
from run_training import Tee                        # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    config = resolve_config(sys.argv[1])
    stage = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'validate'
    extra = [a for a in sys.argv[2:] if a != stage]
    out = os.path.join(json.load(open(config, encoding='utf-8'))['output_dir'], 'tc_gate')
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, f'gate_log_{stage}.txt'), 'a', encoding='utf-8', buffering=1)
    screen = sys.__stdout__
    try:
        screen.reconfigure(encoding='utf-8')
    except Exception:
        pass
    sys.stdout = sys.stderr = Tee(screen, log)
    print(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} 台风期门控 {stage}：{config}', flush=True)
    from wavecast.tc_gate import main as gate
    t0 = time.time()
    gate(['--config', config, '--stage', stage] + extra)
    print(f'=== 完成，用时 {(time.time() - t0) / 60:.1f} 分钟；结果在 {out}', flush=True)


if __name__ == '__main__':
    main()
