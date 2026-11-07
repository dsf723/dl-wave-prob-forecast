# 数据接口约定（UTC；每个站点独立目录）

数据列名区分大小写。缺测用空值，不用 -999 或 0 代替。所有时间建议 ISO8601，如 `2021-08-01T00:00:00Z`。obs 与 forecast 分开存储，事后标签不能写入未来预测特征。

## observations.csv

一行＝站点×有效时间。必须覆盖完整的过去 7 天和未来 14 天目标，默认 3 小时网格；可从原始逐时资料采样，而不是把缺测真值插值为观测。

| 列 | 含义及单位 |
|---|---|
| site_id | A、B、C 或正式装置标识 |
| valid_time | 观测/再分析有效时间 |
| available_time | 这一行历史预测特征实际可获取的时间；再分析应填真实可获取时间，不能填有效时间冒充可实时获取 |
| hs | 有效波高 m；主任务必填 |
| te | 能量周期 s，必须核实谱矩定义 |
| mwd | 平均波浪来向，真北 0°、顺时针 |
| u10, v10 | 10 m 东向、北向风分量 m/s |
| t2m | 2 m 气温 K |
| msl | 海平面气压 Pa |
| label_tc_distance_km | **诊断专用**，由事后最佳路径计算的同一时次最近气旋距离 km |
| event_id | **诊断专用**，整场台风/寒潮统一 ID；跨年事件不能拆成两个 ID |
| tc_distance_km | 作为历史模型输入时，须来自当时可用分析/公告，不能直接复制事后最佳路径 |
| tc_bearing_sin, tc_bearing_cos | 从装置看向气旋中心的方位 sin/cos |
| tc_vmax_ms, tc_speed_ms | 台风最大风速、移动速度 m/s；须注明原资料风速平均时段 |
| tc_probability, cold_probability | 可选历史天气信息；不能填写未来观测标签 |

`label_tc_distance_km`、`event_id` 不属于模型输入，不受历史输入可用时间约束。若无单独 label 距离，诊断会回退到 `tc_distance_km`。确认当时区域无气旋时，可使用 20000 km 表示无近距离影响；资料未取得必须留空，不能假设无台风。`available_time` 应取该行所有实际输入字段的最晚可用时间，或预先保守对齐。最佳路径和现场观测有后续修订时，保留版本。

## forecasts.csv

一行＝site_id×issue_time×valid_time，禁止重复键。

| 列 | 含义及单位 |
|---|---|
| site_id | 对应站点 |
| issue_time | 模式初始化时刻，不等同于下载/发布时刻 |
| available_time | 本行全部预报特征完整可用的最晚时间；应有原始采集/发布记录 |
| valid_time | 未来预报有效时刻 |
| hs, hs_spread | 集合波高均值、标准差，m |
| te, mwd | 核实定义后的周期均值 s、圆周平均来向 ° |
| u10, v10, t2m, msl | 未来预报风温压，单位同上 |
| tc_distance_km, tc_bearing_sin, tc_bearing_cos | **该起报可获得的预报路径**对应的未来位置，不能用最佳路径 |
| tc_vmax_ms, tc_speed_ms | 同次起报的强度和移动速度预报 |
| tc_probability | 集合成员中气旋中心距装置小于预定半径的比例；不能以“无气旋检出”直接替代缺报 |
| cold_probability | 成员预报满足冻结寒潮条件的比例 |
| q000 … q098 | 可选原始集合目标分位数，概率水平为 (i+0.5)/99；用于原始集合概率基线 |
| n_members, source | 推荐保留有效成员数与来源 |

不同预报源拼接时，用截至决策时间可获得的产品，并记录每源初始化、发布时间和模式版本；本行 available_time 取最晚者。插值只能在同次已知预报时次之间进行，不能拿之后新起报的数据回填此前起报。方向用 sin/cos，严禁线性插值 350° 与 10°。

降温 24 小时变化由各样本的“可用历史＋该次未来预报”计算，不读取未来真值。事件路径的插值间隔、风速平均时段、气旋筛选范围需随数据写入来源说明。

## manifest.json

```json
{
  "mode": "reanalysis_proxy",
  "availability_verified": false,
  "site_id": "A",
  "site_lat": 30.5,
  "site_lon": 123.0,
  "observation_source": "ERA5 final, or named in-situ source",
  "forecast_source": "named model, version, archive",
  "period_definition": "verified definition or omitted",
  "availability_provenance": "path to collection/publishing records",
  "track_source": "best track for labels; issued forecast tracks for predictors",
  "units_and_direction": "m, s, K, Pa, m/s; directions from true north",
  "downloaded_at": "actual UTC timestamp",
  "notes": "actual coverage, missingness, interpolation and quality controls"
}
```

只有时间可用性可核实且历史输入满足实时约束时，才使用 `mode=real`、`availability_verified=true`。manifest 不会自动证明资料真实，需要保留采集和版本证据。

## 预测原点

`decision_time=issue_time+decision_delay_hours`，历史输入截至该时刻，目标和未来输入从其后第一步开始。所有 forecast.available_time 必须不晚于 decision_time。超时、缺目标、缺未来预报、跨数据集边界等起报将被排除，原因写入 data_audit.json。特征允许局部缺测并附掩码；正式消融若某特征在训练集中全缺失，会直接报错。
