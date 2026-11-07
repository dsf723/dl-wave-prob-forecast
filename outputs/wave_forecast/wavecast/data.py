"""Normalized CSV contract; observation labels never enter future predictors."""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


def utc(s):
    return pd.to_datetime(s, utc=True, errors="raise", format="mixed")


def read_tables(cfg):
    root = Path(cfg["data_dir"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["mode"] != cfg["mode"]:
        raise ValueError("Config mode and manifest mode disagree")
    if cfg["mode"] not in {"synthetic", "real", "reanalysis_proxy", "pseudo_forecast", "perfect_forecast", "reforecast"}:
        raise ValueError("Unknown experiment mode")
    obs = pd.read_csv(root / "observations.csv")
    fc = pd.read_csv(root / "forecasts.csv")
    if cfg.get("member_features"):
        # each ensemble member's Hs, kept apart from forecasts.csv (prepare_member_columns.py)
        fc = fc.merge(pd.read_csv(root / "members.csv.gz"), on=["issue_time","valid_time"], how="left")
    for df in (obs, fc):
        df.drop(df.index[df.site_id.astype(str) != str(cfg["site_id"])], inplace=True)
        if df.empty:
            raise ValueError("No rows for requested site")
    obs["valid_time"] = utc(obs.valid_time)
    obs = obs.set_index("valid_time").sort_index()
    fc["valid_time"], fc["issue_time"] = utc(fc.valid_time), utc(fc.issue_time)
    if obs.index.duplicated().any() or fc.duplicated(["issue_time", "valid_time"]).any():
        raise ValueError("Duplicate observation or forecast time keys")
    for df in (obs, fc):
        if "available_time" not in df:
            raise ValueError("available_time required; do not assume model initialization equals publication")
        df["available_time"] = utc(df.available_time)
        if df.available_time.isna().any():
            raise ValueError("Missing publication/availability times")
    if (fc.available_time < fc.issue_time).any() or (obs.available_time < obs.index).any():
        raise ValueError("Impossible availability time")
    if cfg["mode"] == "real" and not manifest.get("availability_verified", False):
        raise ValueError("Real mode requires verified availability provenance")
    if not all(k in obs for k in [cfg["target"], "hs"]):
        raise ValueError("Observation target and hs are required")
    if cfg["target"] not in fc:
        raise ValueError("Forecast target mean required for baseline")
    for df in (obs,fc):
        for col in {"hs",cfg["target"]}:
            if col in df and ((df[col].dropna()<0).any() or np.isinf(df[col].dropna()).any()):
                raise ValueError(f"Invalid nonnegative target values in {col}")
    return obs, fc, manifest


def derive(df):
    x = df.copy()
    idx = x.index
    x["doy_sin"] = np.sin(2*np.pi*(idx.dayofyear-1)/365.25)
    x["doy_cos"] = np.cos(2*np.pi*(idx.dayofyear-1)/365.25)
    if "mwd" in x:
        x["wave_sin"], x["wave_cos"] = np.sin(np.deg2rad(x.mwd)), np.cos(np.deg2rad(x.mwd))
    if "u10" in x and "v10" in x:
        x["wind_speed"] = np.hypot(x.u10, x.v10)
    if "t2m" in x:
        lag = x.t2m.reindex(idx - pd.Timedelta(hours=24)).to_numpy()
        x["temp_change_24h"] = x.t2m.to_numpy() - lag
    return x


def season(month, cfg):
    if month in cfg["winter_months"]:
        return "winter_monsoon"
    if month in cfg["summer_months"]:
        return "summer_monsoon"
    return "transition"


def diagnostic_labels(row, cfg):
    dist = row.get("label_tc_distance_km", row.get("tc_distance_km", np.nan))
    tc = pd.notna(dist) and dist <= cfg["typhoon_radius_km"]
    u, v = row.get("u10", np.nan), row.get("v10", np.nan)
    speed = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    delta = row.get("temp_change_24h", np.nan)
    met_known = np.isfinite(speed) and np.isfinite(delta)
    north = min(direction, 360-direction) <= cfg["cold_north_sector_deg"]
    cold = met_known and north and speed >= cfg["cold_wind_ms"] and delta <= -cfg["cold_drop_24h_k"]
    if tc and cold:
        weather = "typhoon_and_cold"
    elif tc:
        weather = "typhoon"
    elif cold:
        weather = "cold_surge"
    elif pd.isna(dist) or not met_known:
        weather = "unknown"
    elif speed <= cfg["calm_wind_ms"] and row.hs < cfg["calm_hs_m"]:
        weather = "calm"
    else:
        weather = "other"
    edges = cfg["wave_bins_m"]
    i = max(0, np.searchsorted(edges, row.hs, side="right") - 1)
    band = f"{edges[i]:g}-{edges[i+1]:g}m" if i+1 < len(edges) else f">={edges[-1]:g}m"
    return weather, band, bool(tc), bool(cold)


def feature_names(cfg, ablation):
    names = [cfg["target"]]
    # windfield (upwind sector winds from a regional wind field) is optional and sits between weather and extreme.
    # tctrack (TIGGE ensemble cyclone-track forecasts, future window only) is optional and follows season.
    for stage in ["base", "season", "tctrack", "weather", "windfield", "extreme"]:
        if stage not in cfg["features"]:
            if stage == ablation:
                raise ValueError(f"Ablation {ablation} has no feature list in config")
            continue
        names += cfg["features"][stage]
        if stage == ablation:
            break
    else:
        raise ValueError(f"Unknown ablation {ablation}")
    return list(dict.fromkeys(names))


def raw_array(df, names):
    return df.reindex(columns=names).to_numpy(dtype=np.float32)


def lead_band(hours_after_first, horizon_days):
    """Lead-time band label: D1-3, D4-7, D8-<horizon> (D8-14 for the original 14-day setup)."""
    return "D1-3" if hours_after_first < 72 else ("D4-7" if hours_after_first < 168 else f"D8-{horizon_days}")


def build_samples(obs, forecasts, cfg, names):
    step = cfg["step_hours"]
    if 24 % step or not 1 <= cfg["horizon_days"] <= 16:
        raise ValueError("step_hours must divide 24; horizon_days must be 1-16 (reforecast files hold 16 days)")
    h, l = cfg["history_days"]*24//step, cfg["horizon_days"]*24//step
    sets = {s: set(cfg[s + "_years"]) for s in ["train", "val", "test"]}
    if any(sets[a] & sets[b] for a, b in [("train","val"),("train","test"),("val","test")]):
        raise ValueError("Split years overlap")
    obs_features = derive(obs)
    samples, skipped = {s: [] for s in sets}, {}
    for issue, group in forecasts.groupby("issue_time", sort=True):
        decision = issue + pd.Timedelta(hours=cfg["decision_delay_hours"])
        pi = pd.date_range(end=decision, periods=h, freq=f"{step}h")
        fi = pd.date_range(start=decision+pd.Timedelta(hours=step), periods=l, freq=f"{step}h")
        years = set(pi.year) | set(fi.year)
        split = next((s for s, allowed in sets.items() if years <= allowed), None)
        if split is None:
            skipped["split_boundary_or_unselected_year"] = skipped.get("split_boundary_or_unselected_year", 0)+1
            continue
        past_source = derive(obs.loc[obs.available_time <= decision]) if cfg["mode"] == "real" else obs_features
        past, truth = past_source.reindex(pi), obs_features.reindex(fi)
        future = group.set_index("valid_time").sort_index().reindex(fi)
        if past[cfg["target"]].isna().any() or truth[cfg["target"]].isna().any() or truth.hs.isna().any():
            skipped["missing_context_or_truth"] = skipped.get("missing_context_or_truth", 0)+1
            continue
        if future[cfg["target"]].isna().any() or future.available_time.isna().any():
            skipped["missing_forecast"] = skipped.get("missing_forecast", 0)+1
            continue
        if (future.available_time > decision).any():
            skipped["forecast_unavailable"] = skipped.get("forecast_unavailable", 0)+1
            continue
        # Proxy modes explicitly permit historical reanalysis context, never future truth as predictors.
        if cfg["mode"] == "real" and (past.available_time > decision).any():
            skipped["context_unavailable"] = skipped.get("context_unavailable", 0)+1
            continue
        # Remove late observations feature-by-feature from real historical inputs.
        # All per-row fields must share the row's verified availability timestamp.
        future_features = derive(pd.concat([past, future], axis=0)).reindex(fi)
        metadata = []
        for j, (t, row) in enumerate(truth.iterrows()):
            weather, band, tc, cold = diagnostic_labels(row, cfg)
            eid = row.get("event_id", "")
            eid = "" if pd.isna(eid) else str(eid)
            if weather in {"typhoon", "cold_surge", "typhoon_and_cold"} and not eid:
                # Unknown event identity: counted as zero independent extreme events.
                block = f"unknown_event_block_{t.strftime('%Y-%W')}"
            else:
                block = eid or f"background_{t.value // pd.Timedelta(days=28).value}"
            metadata.append(dict(site_id=cfg["site_id"], issue_time=issue, decision_time=decision,
                                 valid_time=t, lead_hours=(j+1)*step, lead_day=j*step//24+1,
                                 lead_band=lead_band(j*step, cfg["horizon_days"]),
                                 season=season(t.month,cfg), weather=weather, wave_band=band,
                                 typhoon=tc, cold_surge=cold, event_id=eid, block_id=block,
                                 hs_truth=float(row.hs)))
            forecast_row = future_features.iloc[j]
            # Routing depends only on information available at decision time.
            distance = forecast_row.get("tc_distance_km", np.nan)
            if forecast_row.get("tc_probability", 0) >= .5 or (pd.notna(distance) and distance <= cfg["typhoon_radius_km"]):
                regime = "forecast_typhoon"
            elif forecast_row.get("cold_probability", 0) >= .5:
                regime = "forecast_cold"
            else:
                regime = "forecast_background_or_unknown"
            metadata[-1]["routing_regime"] = regime
        samples[split].append(dict(past=raw_array(past,names), future=raw_array(future_features,names),
                                   y=truth[cfg["target"]].to_numpy(np.float32),
                                   # observed energy period (second target, may be all-NaN for a site without spectra)
                                   y_te=(truth["te_obs"].to_numpy(np.float32) if cfg.get("te_target") and "te_obs" in truth
                                         else None),
                                   raw_mean=future[cfg["target"]].to_numpy(np.float32),
                                   persistence=float(past[cfg["target"]].iloc[-1]), metadata=metadata,
                                   raw_quantiles=(future.reindex(columns=[f"q{k:03d}" for k in range(99)]).to_numpy(np.float32)
                                                  if len(future.filter(regex=r"^q\d{3}$").columns)==99 else None)))
    # An entire named extreme event may belong to only one split.
    events = {s: {m["event_id"] for a in v for m in a["metadata"] if m["event_id"]} for s,v in samples.items()}
    if any(events[a]&events[b] for a,b in [("train","val"),("train","test"),("val","test")]):
        raise ValueError("Named weather event crosses data splits")
    return samples, skipped


class Scaler:
    def fit(self, samples):
        arr = np.concatenate([x[k] for x in samples for k in ["past","future"]])
        count = np.isfinite(arr).sum(0)
        self.mean = np.nansum(np.where(np.isfinite(arr),arr,np.nan),axis=0)/np.maximum(count,1)
        safe = np.where(np.isfinite(arr),arr,self.mean)
        self.std = np.maximum(safe.std(0), 1e-3)
        self.all_missing = np.flatnonzero(count==0).tolist()
        return self

    def transform(self, arr):
        mask = np.isfinite(arr)
        values = (np.where(mask,arr,self.mean)-self.mean)/self.std
        return np.concatenate([values,mask.astype(np.float32)],axis=-1).astype(np.float32)

    def state(self):
        return dict(mean=self.mean.tolist(),std=self.std.tolist(),all_missing=self.all_missing)

    @staticmethod
    def from_state(st):
        if st.get("member_pca"):
            raise NotImplementedError("Member-PCA scaler was only used for validation comparisons")
        sc = PCAScaler(st["pca"]["config"], st["pca"]["names"]) if st.get("pca") else Scaler()
        sc.mean, sc.std = np.array(st["mean"]), np.array(st["std"])
        if st.get("pca"):
            sc.cols, sc.other = st["pca"]["cols"], st["pca"]["other"]
            sc.components, sc.mask_cols = np.array(st["pca"]["components"]), st["pca"]["mask_cols"]
        return sc


class PCAScaler(Scaler):
    """Standardise, then project the physical inputs onto their leading principal components (fitted on the
    training samples only, past and future steps pooled). Site one-hot columns pass through unchanged; a missing
    value is filled with the training mean before projection and flagged by a mask column, kept only for inputs
    that are ever missing in training. cfg: {"components": k} or {"variance": 0.95} (smallest k reaching it);
    optional "passthrough": inputs kept as they are, outside the PCA; optional "mute": inputs blanked at transform
    time (treated as missing), for testing what those inputs contribute."""

    def __init__(self, cfg, names):
        self.cfg, self.names = cfg, list(names)

    def fit(self, samples):
        super().fit(samples)
        arr = np.concatenate([x[k] for x in samples for k in ["past","future"]])
        z = (np.where(np.isfinite(arr),arr,self.mean)-self.mean)/self.std
        keep = set(self.cfg.get("passthrough",[]))
        self.other = [i for i,n in enumerate(self.names) if n.startswith("site_") or n in keep]
        self.mute = [self.names.index(n) for n in self.cfg.get("mute",[])]
        self.cols = [i for i in range(len(self.names)) if i not in self.other]
        _, s, vt = np.linalg.svd(z[:,self.cols]-z[:,self.cols].mean(0), full_matrices=False)
        self.explained = (s**2/np.sum(s**2)).tolist()
        k = self.cfg.get("components") or int(np.searchsorted(np.cumsum(self.explained), self.cfg.get("variance",1.)-1e-9)+1)
        self.components = vt[:k].T
        # Tests of which direction matters: leave out one chosen component, or one component with some inputs zeroed in it.
        if "drop_pc" in self.cfg:
            self.components = np.delete(vt, self.cfg["drop_pc"]-1, axis=0).T
        if "null_from_pc" in self.cfg:
            n = vt[self.cfg["null_from_pc"]-1].copy()
            n[[[self.names[i] for i in self.cols].index(x) for x in self.cfg["null_zero"]]] = 0
            n /= np.linalg.norm(n)
            self.components = np.linalg.svd(np.eye(len(n))-np.outer(n,n))[0][:,:len(n)-1]
        self.mask_cols = np.flatnonzero(~np.isfinite(arr).all(0)).tolist()
        return self

    def transform(self, arr):
        if getattr(self,"mute",None):
            arr = arr.copy()
            arr[:,self.mute] = np.nan
        mask = np.isfinite(arr)
        z = (np.where(mask,arr,self.mean)-self.mean)/self.std
        return np.concatenate([z[:,self.cols]@self.components, z[:,self.other],
                               mask[:,self.mask_cols].astype(np.float32)],axis=-1).astype(np.float32)

    def state(self):
        return dict(super().state(), pca=dict(config=self.cfg, names=self.names, cols=self.cols, other=self.other,
                                              components=self.components.tolist(), mask_cols=self.mask_cols,
                                              explained=getattr(self,"explained",None)))


class MemberPCAScaler(Scaler):
    """Replace the ensemble mean and spread by the leading principal components of the members' Hs (m00-m10).
    The components are fitted on the future steps of the training samples; the past window, which has observed Hs but
    no members, gets zeros and a mask. Future Hs (the ensemble mean) is masked, so the members reach the model only
    through the components; hs_spread is dropped. cfg: {"components": k}."""

    def __init__(self, cfg, names):
        self.cfg, self.names = cfg, list(names)

    def fit(self, samples):
        super().fit(samples)
        self.cols = [i for i,n in enumerate(self.names) if re.fullmatch(r"m\d\d", n)]
        self.hs, drop = self.names.index("hs"), self.names.index("hs_spread")
        self.other = [i for i in range(len(self.names)) if i not in self.cols and i != drop]
        fut = np.concatenate([x["future"] for x in samples])[:,self.cols]
        self.mean[self.cols] = np.nanmean(fut,axis=0)
        self.std[self.cols] = np.maximum(np.nanstd(fut,axis=0),1e-3)
        z = np.nan_to_num((fut-self.mean[self.cols])/self.std[self.cols])
        _, s, vt = np.linalg.svd(z-z.mean(0), full_matrices=False)
        self.explained = (s**2/np.sum(s**2)).tolist()
        self.components = vt[:self.cfg["components"]].T
        return self

    def transform(self, arr):
        mask = np.isfinite(arr)
        z = (np.where(mask,arr,self.mean)-self.mean)/self.std
        return np.concatenate([z[:,self.other], z[:,self.cols]@self.components,
                               mask[:,[self.hs]].astype(np.float32), mask[:,self.cols].all(1,keepdims=True).astype(np.float32)],
                              axis=-1).astype(np.float32)

    def transform_future(self, arr):
        arr = arr.copy()
        arr[:,self.hs] = np.nan
        return self.transform(arr)

    def state(self):
        return dict(super().state(), member_pca=dict(config=self.cfg, names=self.names, components=self.components.tolist(), explained=self.explained))


def make_scaler(cfg, names):
    if cfg.get("member_pca"):
        return MemberPCAScaler(cfg["member_pca"], names)
    return PCAScaler(cfg["pca"], names) if cfg.get("pca") else Scaler()
