# 大浪上尾重训对照（10-27）：本机基准 + 3 个方案，依次在验证集上训练（中断后重跑同一命令可续跑）
$configs = @('GKH_h15_local_transformer_pretrain', 'GKH_h15_tw3_transformer_pretrain',
             'GKH_h15_tw2thr_transformer_pretrain', 'GKH_h15_deg24_transformer_pretrain')
foreach ($c in $configs) {
    Write-Host "===== $c =====" -ForegroundColor Cyan
    python run_training.py "model_configs/$c.json" validate
}
Write-Host '===== 全部完成，对比：python compare_tail_variants.py =====' -ForegroundColor Green
