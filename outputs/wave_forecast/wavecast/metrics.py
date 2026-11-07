"""Scores per forecast case, grouped diagnosis and paired cluster bootstrap."""
import numpy as np
import pandas as pd

TAU = (np.arange(99)+.5)/99


def score(q, y):
    error = np.asarray(y)[...,None] - np.asarray(q)
    return 2*np.maximum(TAU*error,(TAU-1)*error).mean(-1)


def quantile_at(q, p):
    return np.asarray([np.interp(p,TAU,row) for row in q])


def case_table(samples, q, label, cfg, climatology):
    rows = pd.DataFrame([m for s in samples for m in s["metadata"]])
    y = np.concatenate([s["y"] for s in samples])
    rows["model"], rows["truth"] = label, y
    rows["crps"] = score(q,y)
    rows["clim_crps"] = score(climatology,y)
    for p,name in [(.1,"p10"),(.5,"p50"),(.9,"p90")]:
        rows[name] = quantile_at(q,p)
    rows["covered80"] = (y>=rows.p10)&(y<=rows.p90)
    rows["width80"] = rows.p90-rows.p10
    rows["error50"] = rows.p50-y
    if cfg["target"] == "hs":
        threshold = cfg["extreme_hs_m"]
        prob = np.asarray([1-np.interp(threshold,row,TAU,left=0,right=1) for row in q])
        rows["exceed_probability"] = prob
        rows["exceed_brier"] = (prob-(y>=threshold))**2
    return rows


def grouped(table, cfg):
    groups = [[],["lead_day"],["season"],["weather"],["wave_band"],
              ["weather","lead_day"],["season","lead_day"],["wave_band","lead_day"],
              ["weather","wave_band"],["season","weather","lead_day"]]
    if "site_id" in table and table.site_id.nunique()>1:
        groups += [["site_id"],["site_id","weather"],["site_id","lead_day"]]
    records=[]
    for model, sub in table.groupby("model"):
        for keys in groups:
            iterator = sub.groupby(keys,dropna=False) if keys else [((),sub)]
            for key, g in iterator:
                vals = key if isinstance(key,tuple) else (key,)
                n_events = g.loc[g.event_id.fillna("").astype(str).ne(""),"event_id"].nunique()
                ref = g.clim_crps.mean()
                rec = dict(model=model,group="×".join(keys) or "overall",n_cases=len(g),
                           n_valid_times=g.valid_time.nunique(),n_events=n_events,
                           n_blocks=g.block_id.nunique(),crps=g.crps.mean(),
                           crpss=1-g.crps.mean()/ref if ref>1e-12 else np.nan,
                           coverage80=g.covered80.mean(),width80=g.width80.mean(),
                           bias50=g.error50.mean(),mae50=g.error50.abs().mean(),
                           rmse50=np.sqrt((g.error50**2).mean()))
                rec.update(dict(zip(keys,vals)))
                rec["insufficient_extreme_events"] = bool(g.typhoon.any() or g.cold_surge.any()) and n_events<cfg["min_group_events"]
                if "exceed_brier" in g:
                    rec["exceed_brier"] = g.exceed_brier.mean()
                records.append(rec)
    return pd.DataFrame(records)


def paired_bootstrap(a,b, repeats=1000, seed=61):
    """Forecast-case weighted mean difference, resampling entire weather/time blocks."""
    keys=["site_id","issue_time","valid_time"]
    x=a[keys+["block_id","crps"]].merge(b[keys+["crps"]],on=keys,validate="one_to_one",suffixes=("_a","_b"))
    x["difference"]=x.crps_a-x.crps_b
    sums=x.groupby("block_id").difference.agg(["sum","count"])
    if len(sums)<2:
        return dict(delta=float(x.difference.mean()),ci_low=None,ci_high=None,n_blocks=len(sums))
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(sums),size=(repeats,len(sums)))
    boot=sums["sum"].to_numpy()[ix].sum(1)/sums["count"].to_numpy()[ix].sum(1)
    return dict(delta=float(x.difference.mean()),ci_low=float(np.quantile(boot,.025)),
                ci_high=float(np.quantile(boot,.975)),n_blocks=len(sums))


def event_diagnostics(table,cfg):
    records=[]
    extremes=table[table.event_id.fillna("").astype(str).ne("")]
    for (model,issue,event),g in extremes.groupby(["model","issue_time","event_id"]):
        g=g.sort_values("valid_time")
        truth_peak=g.loc[g.truth.idxmax()]
        predicted_peak=g.loc[g.p50.idxmax()]
        rec=dict(model=model,issue_time=issue,event_id=event,n_cases=len(g),
                 observed_peak=truth_peak.truth,predicted_peak=predicted_peak.p50,
                 peak_bias=predicted_peak.p50-truth_peak.truth,
                 peak_time_error_hours=(pd.Timestamp(predicted_peak.valid_time)-pd.Timestamp(truth_peak.valid_time)).total_seconds()/3600,
                 coverage80=g.covered80.mean(),crps=g.crps.mean(),
                 note="within available event window; boundary peaks may be truncated")
        if cfg["target"]=="hs":
            rec["observed_exceed_hours"]=(g.truth>=cfg["extreme_hs_m"]).sum()*cfg["step_hours"]
            rec["predicted_exceed_hours_p50"]=(g.p50>=cfg["extreme_hs_m"]).sum()*cfg["step_hours"]
        records.append(rec)
    return pd.DataFrame(records)
