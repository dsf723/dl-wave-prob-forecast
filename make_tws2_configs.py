"""台湾海峡扩充训练配置（11-04 事先登记的第 1 条）：model_configs/TWS2_h15_transformer_pretrain.json、TWS2_h15_gru_pretrain.json。

从冻结的 TWS_h15 两个配置复制，超参数、损失、输入都不改，不调参；只换站点和年份。
站点：新竹、马祖改用气象署历史资料（CW46757B、CWC6W08，2000 年起 / 2010-08 起）；澎湖、金门、七股照旧（TW*，Copernicus 2012-09 至 2017-03）。
年份：训练 2000–2015（各站从有浮标起），验证 2016–2019（TW* 只到 2017-03）；测试 2020-09-23 至 2024-12-31 的业务 GEFS，
      在 evaluate_operational.py 里与冻结模型一次性评分，这里的 test_years 只是占位（回算没有 2020 年以后）。
预训练（ERA5 临时真值）：马祖 2000–2009；澎湖、金门、七股 2000–2011（沿用 site_TW*_era5pre）；新竹不预训练（1997 年起就有浮标）。
用法：python make_taiwan_strait_configs.py 之后运行 python make_tws2_configs.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = 'F:/Offshore Power Forecasting'
TRAIN = {'CW46757B': range(2000, 2016), 'CWC6W08': range(2010, 2016), 'TW46735A': range(2012, 2016), 'TW46787A': range(2012, 2016), 'TW46778A': range(2012, 2016)}
VAL = {'CW46757B': range(2016, 2020), 'CWC6W08': range(2016, 2020), 'TW46735A': range(2016, 2018), 'TW46787A': range(2016, 2018), 'TW46778A': range(2016, 2018)}
TEST = list(range(2020, 2025))
PRE = {'CWC6W08': range(2000, 2010), 'TW46735A': range(2000, 2012), 'TW46787A': range(2000, 2012), 'TW46778A': range(2000, 2012)}


def main():
    for model in ('transformer', 'gru'):
        c = json.load(open(os.path.join(ROOT, 'model_configs', f'TWS_h15_{model}_pretrain.json'), encoding='utf-8'))
        name = f'TWS2_h15_{model}_pretrain'
        c.update(data_dir=f'{BASE}/processed/site_CW46757B_reforecast', output_dir=f'{BASE}/model_runs/{name}', site_id='TWS2',
                 train_years=list(TRAIN['CW46757B']), val_years=list(VAL['CW46757B']), test_years=TEST,
                 sites=[dict(site_id=s, data_dir=f'{BASE}/processed/site_{s}_reforecast', train_years=list(TRAIN[s]), val_years=list(VAL[s]), test_years=TEST) for s in TRAIN])
        c['pretrain'] = dict(epochs=c['pretrain']['epochs'],
                             sites=[dict(site_id=s, data_dir=f'{BASE}/processed/site_{s}_era5pre', train_years=list(y), val_years=[], test_years=[]) for s, y in PRE.items()])
        path = os.path.join(ROOT, 'model_configs', f'{name}.json')
        json.dump(c, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(path)


if __name__ == '__main__':
    main()
