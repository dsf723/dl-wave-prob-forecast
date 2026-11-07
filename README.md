# Probabilistic medium-range wave forecasting by deep-learning post-processing of GEFSv12

This repository contains the code, model configurations and frozen model weights of a post-processing system for the
NOAA GEFSv12 wave ensemble. For each buoy site it issues 99 quantiles of significant wave height (Hs) every 3 h out to
15 days, extends them to a 30-day product with climatology for days 16–30, and derives the energy period, wave power and
wave-energy resource indices.

中文简介：基于深度学习后处理 GEFSv12 海浪集合的有效波高中期概率预报。Transformer 与 GRU 编码器 + Bernstein 分位数输出，
以 CRPS + 阈值加权 CRPS（2.5 m）训练，两模型分位数平均；第 1–15 天为模型预报，第 16–30 天为气候态。

## Method in brief

* **Inputs** (per 3-hourly valid time, days 0–15): ensemble mean and spread of Hs, energy period Te, wave direction,
  day-of-year terms and a one-hot site code; the past 7 days of buoy Hs and ERA5 Te / direction.
* **Encoders**: Transformer and GRU, three seeds each, trained jointly on all sites of a region.
* **Output**: Bernstein-polynomial quantile function (99 monotone quantiles).
* **Loss / selection score**: CRPS + 1 × twCRPS (threshold 2.5 m).
* **Training**: 20 epochs of pre-training with ERA5 as proxy truth, then fine-tuning on buoy observations with early stopping.
* **Ensemble**: quantile averaging (Vincentization) of Transformer and GRU, weight 0.5.
* **Benchmarks**: raw GEFS ensemble, EMOS (truncated normal and log-normal), monthly climatology, persistence.
* **Evaluation**: once on a 2017–2019 reforecast test set and once on 2020–2024 operational GEFS forecasts, with a paired
  block bootstrap (events and 28-day blocks, 2000 resamples).

Main sites: Guam (NDBC 52200), Marado (KMA 22107), Hawaii (NDBC 51207). Transfer test: five Taiwan Strait buoys.

| Data set | Ensemble CRPS (m) | vs raw GEFS | vs EMOS (log-normal) | vs climatology | P10–P90 coverage |
|---|---:|---:|---:|---:|---:|
| Reforecast test 2017–2019 | 0.2272 | −24.5% | −13.4% | −28.4% | 82.0% |
| Operational GEFS 2020–2024 | 0.2236 | −21.8% | −8.8% | −28.5% | 81.9% |

## Repository layout

```
outputs/wave_forecast/wavecast/   core package: data handling, models (Transformer, GRU, TiDE, CNN-BiGRU,
                                  Bernstein head), metrics (CRPS, twCRPS, bootstrap), EMOS, experiment runner
outputs/wave_forecast/tests/      unit tests
model_configs/                    experiment configurations (final runs: GKH_h15_*_pretrain, TWS_h15_*_pretrain)
model_runs/<run>/                 frozen weights (*.pt, 3 seeds), config_used.json, selection.json (config fingerprints)
metadata/sites.json               site definitions; gefs_reforecast_era5_ratios.json: period/height conversion factors
download_*.py                     data download (GEFSv12 reforecast and operational, ERA5, buoys, altimeter, tracks)
prepare_*.py, era5_to_observations.py, label_typhoons.py   dataset preparation
run_training.py                   train / validate / test a configuration
evaluate_operational.py           independent evaluation of frozen models on operational GEFS
forecast_now.py                   one-command real-time 30-day forecast
export_quantile_forecasts.py, forecast_monthly_energy.py, forecast_intramonth_variability.py, joint_scenarios.py
                                  30-day product, wave power and intra-month variability
site_resource_metrics.py, historical_wave_energy.py   wave-energy resource (mean power, MVI, CV)
tune_*.py, prove_*.py, compare_*.py, selection_rule_check.py   tuning, weight tests and model comparison
make_manuscript_figs.py           figures of the manuscript
```

Most comments and docstrings are in Chinese.

## Requirements

Python 3.11+ and the packages in `requirements.txt`. GPU is optional (training of one configuration takes about
10 min on a single GPU). Downloading ERA5 requires a Copernicus Climate Data Store account (`~/.cdsapirc`), and the
altimeter download requires `copernicusmarine login`; no credentials are stored in this repository.

```bash
pip install -r requirements.txt
python -m pytest outputs/wave_forecast/tests
```

## Data

The observation and forecast data are **not** included, because of their size (about 50 GB) and because several
providers have their own terms of use. All of them are public and can be downloaded with the scripts here:

| Data | Source |
|---|---|
| GEFSv12 wave reforecast (2000–2019) and operational forecasts (2020–) | NOAA, AWS / Google Cloud open data |
| ERA5 single-level waves and winds | Copernicus Climate Data Store |
| Guam and Hawaii buoys | NDBC, CDIP |
| Marado buoy | Korea Meteorological Administration (data.kma.go.kr) |
| Taiwan Strait buoys | Copernicus Marine in-situ product INSITU_GLO_WAV_DISCRETE_MY_013_045; Central Weather Administration, Marine Environmental Information Platform |
| Tropical-cyclone tracks | IBTrACS v04r01 |

Typical order:

```bash
python download_gefs_reforecast.py           # Wednesday 11-member reforecast, site series
python download_buoys.py                      # NDBC / CDIP buoy observations
python prepare_kma_buoy.py                    # Marado buoy files downloaded from data.kma.go.kr
python download_era5_timeseries.py G K H      # ERA5 site series
python era5_to_observations.py G K H
python label_typhoons.py
python prepare_reforecast_dataset.py G K H
python prepare_era5pretrain_dataset.py G K H
python run_training.py model_configs/GKH_h15_transformer_pretrain.json validate
python run_training.py model_configs/GKH_h15_gru_pretrain.json validate
```

Run each script with `-h` for its options. Absolute paths in `model_configs/*.json` are rewritten to the local project
directory at run time (`project_paths.py`).

## Using the frozen models

`model_runs/` holds the five frozen runs used in the paper:

| Run | Sites | Target |
|---|---|---|
| GKH_h15_transformer_pretrain, GKH_h15_gru_pretrain | Guam, Marado, Hawaii | Hs (ensemble members) |
| GKH_h15_te_transformer_pretrain | Guam, Marado, Hawaii | Te (multi-task head) |
| TWS_h15_transformer_pretrain, TWS_h15_gru_pretrain | 5 Taiwan Strait buoys | Hs |

The evaluation scripts refuse to score a run whose configuration differs from the frozen one (`selection.json`,
portable fingerprint). With the prepared data in `processed/`:

```bash
python evaluate_operational.py --a GKH_h15_transformer_pretrain --b GKH_h15_gru_pretrain --suffix _oper --years 2020 2024
python forecast_now.py --date 20261028
```

## License

The code and model weights are released under the MIT License (see `LICENSE`). The observation and forecast data are not covered by this license; they remain subject to the terms of their providers.
