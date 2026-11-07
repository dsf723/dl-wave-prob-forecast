# 10-30：机器空闲时依次单独运行，比较 PCA 与基准的训练用时；最后补跑带天气变量的 PCA
python run_training.py model_configs/GKH_h15_solo_base_transformer_pretrain.json
python run_training.py model_configs/GKH_h15_solo_pca95_transformer_pretrain.json
python run_training.py model_configs/GKH_h15_solo_pcak4_transformer_pretrain.json
python run_training.py model_configs/GKH_h15_solo_base_gru_pretrain.json
python run_training.py model_configs/GKH_h15_solo_pca95_gru_pretrain.json
python run_training.py model_configs/GKH_h15_solo_pcak4_gru_pretrain.json
python run_training.py model_configs/GKH_h15_pcaW95_transformer_pretrain.json
