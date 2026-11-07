# 10-30：PCA 队列完成后，单独跑一遍基准（比较训练用时），再开始第二版贝叶斯优化
while (-not ((Test-Path 'model_runs/GKH_h15_pcaW95_transformer_pretrain/selection.json') -or (Select-String -Path 'model_runs/GKH_h15_pcaW95_transformer_pretrain/train_log_validate.txt' -Pattern 'interrupted|中断' -Quiet -ErrorAction SilentlyContinue))) { Start-Sleep -Seconds 30 }
python run_training.py model_configs/GKH_h15_timing_base_transformer_pretrain.json
python run_training.py model_configs/GKH_h15_timing_base_gru_pretrain.json
python tune_lambda_bo.py transformer
python tune_lambda_bo.py gru
