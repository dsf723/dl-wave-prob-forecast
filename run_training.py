"""训练启动器：运行 wavecast.experiment，输出同时显示在屏幕上并写进结果目录的 train_log_<阶段>.txt。

两种用法：
  1. 在终端（可见窗口）里运行，看得到每一轮的进度（默认）：
       python run_training.py <配置.json> [validate|test]
     屏幕上显示"第几个训练 / 共几个"、每轮 train/val CRPS、每轮耗时，*best 表示验证成绩刷新。
     关闭窗口或按 Ctrl+C 会中断；重新运行同一条命令即可断点续跑（已完成的训练直接读取）。
  （云服务器上同样用法；配置里写的本机路径 F:/Offshore Power Forecasting/... 会自动换成服务器上的项目目录）
  2. 用 pythonw + 计划任务在后台运行（没有窗口时只写日志）：
       pythonw run_training.py <配置.json> [validate|test]
"""
import json
import os
import runpy
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, 'outputs', 'wave_forecast')


class Tee:
    """同时写屏幕和日志文件。"""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):          # 日志库（如 Optuna 的彩色日志）会询问输出是否为终端
        return False


def main():
    sys.path.insert(0, ROOT)
    from project_paths import resolve_config        # 云服务器上自动把本机路径换成项目目录
    config = resolve_config(sys.argv[1])
    stage = sys.argv[2] if len(sys.argv) > 2 else 'validate'
    out = json.load(open(config, encoding='utf-8'))['output_dir']
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, f'train_log_{stage}.txt'), 'a', encoding='utf-8', buffering=1)
    screen = sys.__stdout__
    if screen is not None:
        try:
            screen.reconfigure(encoding='utf-8')
        except Exception:
            pass
    sys.stdout = sys.stderr = Tee(screen, log)
    print(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} 开始 {stage}：{config}', flush=True)
    print(f'    日志：{os.path.join(out, f"train_log_{stage}.txt")}', flush=True)
    os.chdir(PKG)
    sys.path.insert(0, PKG)
    sys.argv = ['wavecast.experiment', '--config', config, '--stage', stage]
    t0 = time.time()
    try:
        runpy.run_module('wavecast.experiment', run_name='__main__')
        print(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} 完成，用时 {(time.time() - t0) / 3600:.1f} 小时', flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        print(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} 中断或失败（重新运行同一条命令可断点续跑）', flush=True)
        raise


if __name__ == '__main__':
    main()
