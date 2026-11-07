"""台湾海峡 5 站配置：model_configs/TWS_h15_transformer_pretrain.json、TWS_h15_gru_pretrain.json。

从 GKH_h15 两个最终配置复制，超参数、损失、输入都不改，不重新调参；只换站点、年份、输出目录。
站点：C6W08 马祖、46757B 新竹、46735A 澎湖、46787A 金门、46778A 七股。保留站点编号（一个模型内 5 个站）。
年份：浮标训练 2012–2014（2012 从 9 月起），验证 2015，测试 2016–2017（到 2017-03）；ERA5 预训练 2000–2011。方案在看到结果前写定。
用法：python make_taiwan_strait_configs.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = 'F:/Offshore Power Forecasting'
SITES = ['TWC6W08', 'TW46757B', 'TW46735A', 'TW46787A', 'TW46778A']
TRAIN, VAL, TEST = [2012, 2013, 2014], [2015], [2016, 2017]
PRE = list(range(2000, 2012))


def main():
    for model in ('transformer', 'gru'):
        c = json.load(open(os.path.join(ROOT, 'model_configs', f'GKH_h15_{model}_pretrain.json'), encoding='utf-8'))
        name = f'TWS_h15_{model}_pretrain'
        c.update(data_dir=f'{BASE}/processed/site_{SITES[0]}_reforecast', output_dir=f'{BASE}/model_runs/{name}', site_id='TWS',
                 train_years=TRAIN, val_years=VAL, test_years=TEST,
                 sites=[dict(site_id=s, data_dir=f'{BASE}/processed/site_{s}_reforecast', train_years=TRAIN, val_years=VAL, test_years=TEST) for s in SITES])
        c['pretrain'] = dict(epochs=c['pretrain']['epochs'],
                             sites=[dict(site_id=s, data_dir=f'{BASE}/processed/site_{s}_era5pre', train_years=PRE, val_years=[], test_years=[]) for s in SITES])
        path = os.path.join(ROOT, 'model_configs', f'{name}.json')
        json.dump(c, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(path)


if __name__ == '__main__':
    main()
