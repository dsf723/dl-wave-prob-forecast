"""ERA5 offline E1/E2 experiments. Assumed errors are not actual forecast errors."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import write_json


def ar1(n,rng,rho=.9):
    z=rng.normal(size=n)
    for i in range(1,n):
        z[i]=rho*z[i-1]+np.sqrt(1-rho*rho)*z[i]
    return z


def generate(observations,directory,perfect=False,seed=82,error_scale=1,origin_stride_days=1,wind_wave_corr=.7):
    # wind_wave_corr: correlation of wave-height error with wind error. Real wave models are driven by the
    # forecast wind, so 0.9-1 is closer to reality; 0.7 (default) leaves a partly independent wave error.
    if error_scale<=0 and not perfect:
        raise ValueError("E2 requires a positive error scale")
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists():
        raise ValueError("Output dataset exists; use another directory")
    obs=pd.read_csv(observations)
    if obs.site_id.nunique()!=1:
        raise ValueError("Prepare one site per dataset")
    obs["valid_time"]=pd.to_datetime(obs.valid_time,utc=True,format="mixed")
    indexed=obs.set_index("valid_time").sort_index()
    if indexed.index.duplicated().any():
        raise ValueError("Duplicate observation times")
    rng=np.random.default_rng(seed)
    rows=[]
    for issue in pd.date_range(indexed.index.min().ceil("D")+pd.Timedelta(days=7),
                               indexed.index.max()-pd.Timedelta(days=14),freq=f"{origin_stride_days}D"):
        idx=pd.date_range(issue+pd.Timedelta(hours=3),periods=112,freq="3h")
        source=indexed.reindex(idx)
        if source.hs.isna().any():
            continue
        # Whitelist predictors; exclude event identities and all diagnostic-only fields.
        columns=[c for c in ["site_id","hs","te","mwd","u10","v10","t2m","msl"] if c in source]
        columns+=[c for c in source.columns if c.startswith("wf_")]          # upwind sector wind features
        f=source[columns].copy()
        day=np.arange(1,113)/8
        sigma=error_scale*np.minimum(.08+.025*day,.5)
        wind_noise=ar1(len(f),rng)
        wave_noise=wind_wave_corr*wind_noise+np.sqrt(1-wind_wave_corr**2)*ar1(len(f),rng)
        if not perfect:
            f["hs"]*=np.exp(sigma*wave_noise-.5*sigma**2)
            if "te" in f:
                period_noise=.5*wave_noise+np.sqrt(.75)*ar1(len(f),rng)
                f["te"]*=np.exp(.4*sigma*period_noise)
            if "mwd" in f:
                f["mwd"]=(f.mwd+error_scale*(5+day)*ar1(len(f),rng))%360
            # Both wind components receive correlated, temporally persistent errors.
            for col in ["u10","v10"]:
                if col in f:
                    f[col]+=error_scale*(.5+.25*day)*(.7*wind_noise+np.sqrt(.51)*ar1(len(f),rng))   # wind_noise is the shared component
            if "t2m" in f:
                f["t2m"]+=error_scale*(.3+.15*day)*ar1(len(f),rng)
            if "msl" in f:
                f["msl"]+=error_scale*(50+20*day)*ar1(len(f),rng)
            # Wind-field features: drawn after all other noise so existing columns stay identical for a given seed.
            # Large-scale wind errors are spatially coherent: one shared series (tied to the site wind error)
            # plus a small block-specific part, so the model cannot average independent noise away.
            wf=[c for c in f.columns if c.startswith("wf_")]
            if wf:
                common=.7*wind_noise+np.sqrt(.51)*ar1(len(f),rng)
                for col in wf:
                    f[col]+=error_scale*(.5+.25*day)*(.9*common+np.sqrt(.19)*ar1(len(f),rng))
        f["hs_spread"]=0 if perfect else sigma*f.hs
        f["issue_time"]=issue; f["available_time"]=issue
        f.index.name="valid_time"
        rows.append(f.reset_index())
    if not rows:
        raise ValueError("No complete 14-day windows; prepare 3-hourly observation grid")
    obs.to_csv(root/"observations.csv",index=False)
    pd.concat(rows,ignore_index=True).to_csv(root/"forecasts.csv",index=False)
    write_json(root/"manifest.json",dict(mode="perfect_forecast" if perfect else "pseudo_forecast",
               availability_verified=False,source=str(observations),seed=seed,error_scale=error_scale,
               origin_stride_days=origin_stride_days,wind_wave_corr=wind_wave_corr,
               note="Future inputs generated offline from reanalysis. E1 ideal reference / E2 assumed correlated errors; no operational skill claim. No future best-track features generated."))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--observations",required=True); p.add_argument("--directory",required=True)
    p.add_argument("--perfect",action="store_true"); p.add_argument("--seed",type=int,default=82)
    p.add_argument("--error-scale",type=float,default=1.)
    p.add_argument("--origin-stride-days",type=int,default=1)
    p.add_argument("--wind-wave-corr",type=float,default=.7)
    a=p.parse_args()
    generate(a.observations,a.directory,a.perfect,a.seed,a.error_scale,a.origin_stride_days,a.wind_wave_corr)
