#!/usr/bin/env bash
# 独立业务检验：GEFS 业务海浪预报按起报年份分成多个 tmux 窗口并行下载（10-27）。
# 每个窗口只写自己年份的 <站号>/<年>.csv.gz，不会互相覆盖；_done.txt 是逐行追加，多进程共用没问题。
# 中断后再运行本脚本即可续传（已完成的起报日会跳过）。
# 用法：bash run_oper_download.sh                    # 2020–2024 五个窗口，每窗口 3 进程 × 12 线程
#       PROCS=4 THREADS=16 bash run_oper_download.sh  # 网速还有富余时加大
#       YEARS="2020 2021 2022 2023 2024 2025" bash run_oper_download.sh   # ERA5 补到 2025 年后
#       MIRROR=gcs bash run_oper_download.sh          # 先取 Google Cloud 镜像（缺的退回 AWS）
#       SITES="CW46757B CWC6W08 ..." OUT=processed/gefs_operational_tws SESSION=gefs_tws bash run_oper_download.sh   # 11-04 台湾海峡（新目录，不受旧 _done.txt 影响）
# 查看：tmux attach -t gefs_oper（Ctrl+b 再按 n / p 切换窗口，Ctrl+b 再按 d 离开且不中断）
SESSION=${SESSION:-gefs_oper}
SITES=${SITES:-"G K H"}
OUT=${OUT:-processed/gefs_operational}
PROCS=${PROCS:-3}
THREADS=${THREADS:-12}
MIRROR=${MIRROR:-aws}
YEARS=${YEARS:-"2020 2021 2022 2023 2024"}
cd "${PROJECT_DIR:-$(pwd)}" || exit 1
PROJECT_DIR=$(pwd)
mkdir -p logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux 会话 $SESSION 已在运行：tmux attach -t $SESSION（要重开先在里面全部结束，或 tmux kill-session -t $SESSION）"
    exit 1
fi
first=1
for y in $YEARS; do
    start="$y-01-01"
    [ "$y" = 2020 ] && start=2020-09-23
    cmd="${ENV_SETUP:+$ENV_SETUP && }cd '$PROJECT_DIR' && python download_gefs_operational.py --start $start --end $y-12-31 --procs $PROCS --threads $THREADS --mirror $MIRROR --sites $SITES --out $OUT 2>&1 | tee -a logs/${SESSION}_$y.log; echo '== $y 结束（窗口保留，exit 关闭）'; exec bash"
    if [ $first = 1 ]; then
        tmux new-session -d -s "$SESSION" -n "y$y" "$cmd"
        first=0
    else
        tmux new-window -t "$SESSION" -n "y$y" "$cmd"
    fi
done
echo "已启动：tmux attach -t $SESSION"
