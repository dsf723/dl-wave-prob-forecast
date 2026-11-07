# 10-30：等 λ 网格最后一个配置（lam4 GRU）训练完，再依次训练 PCA 配置（同一台机器、不并行，训练用时才可比）
Write-Host '等待 λ 网格训练完成……'
while (-not (Test-Path 'model_runs/GKH_h15_lam4_gru_pretrain/selection.json')) { Start-Sleep -Seconds 60 }
python run_training.py model_configs/GKH_h15_pca95_transformer_pretrain.json
python run_training.py model_configs/GKH_h15_pca95_gru_pretrain.json
python run_training.py model_configs/GKH_h15_pcak4_transformer_pretrain.json
python run_training.py model_configs/GKH_h15_pcak4_gru_pretrain.json
python run_training.py model_configs/GKH_h15_pcaW95_transformer_pretrain.json
