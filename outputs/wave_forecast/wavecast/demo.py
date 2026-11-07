"""Synthetic integration test data, intentionally not evidence of forecast skill."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import write_json,run


def generate(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(812)
    idx=pd.date_range("2018-01-01","2021-12-31 21:00",freq="3h",tz="UTC")
    n=len(idx); doy=idx.dayofyear.to_numpy(); hour=idx.hour.to_numpy()
    tc=np.zeros(n); cold=np.zeros(n)
    event=np.full(n,"",dtype=object)
    for year in range(2018,2022):
        for month,day,kind in [(1,18,"cold"),(3,7,"cold"),(7,18,"tc"),(8,22,"tc"),(9,16,"tc"),(12,6,"cold")]:
            center=pd.Timestamp(year=year,month=month,day=day,tz="UTC")
            distance=np.abs((idx-center).total_seconds()/86400)
            pulse=np.maximum(0,1-distance/3)
            if kind=="tc": tc+=pulse
            else: cold+=pulse
            event[distance<=3]=f"synthetic_{kind}_{year}_{month:02d}"
    u=3+2*np.sin(doy/14)+10*tc
    v=-3-2*np.cos(doy/17)-14*cold
    t=289+9*np.sin(2*np.pi*(doy-100)/365)-32*cold
    hs=np.maximum(.1,1.2+.5*np.cos(2*np.pi*doy/365)+5*tc+2*cold+.12*rng.normal(size=n))
    obs=pd.DataFrame(dict(site_id="DEMO",valid_time=idx,available_time=idx,hs=hs,
                          te=5+hs*.6,u10=u,v10=v,t2m=t,msl=101300-3200*tc-500*cold,
                          mwd=(70+30*np.sin(doy/20))%360,tc_distance_km=np.where(tc>0,900*(1-tc),20000),
                          tc_vmax_ms=tc*45,tc_speed_ms=tc*6,tc_bearing_sin=np.sin(doy),
                          tc_bearing_cos=np.cos(doy),tc_probability=(tc>0).astype(float),
                          cold_probability=(cold>0).astype(float),event_id=event,hs_spread=.2))
    obs.to_csv(root/"observations.csv",index=False)
    indexed=obs.set_index("valid_time")
    rows=[]
    for year in range(2018,2022):
        for issue in pd.date_range(f"{year}-01-08",f"{year}-12-16",freq="14D",tz="UTC"):
            valid=pd.date_range(issue+pd.Timedelta(hours=3),periods=112,freq="3h")
            f=indexed.reindex(valid).drop(columns="event_id").copy()
            f.index.name="valid_time"
            lead=np.arange(1,113)/8
            err=rng.normal(size=len(f))*(.1+.04*lead)
            f["hs"]=np.maximum(.05,f.hs*.9+err)
            f["hs_spread"]=.2+.06*lead
            f["te"]+=rng.normal(0,.2,len(f))
            f["tc_distance_km"]=np.maximum(0,f.tc_distance_km+rng.normal(0,70,len(f)))
            f["tc_probability"]=np.clip(f.tc_probability+rng.normal(0,.15,len(f)),0,1)
            f["cold_probability"]=np.clip(f.cold_probability+rng.normal(0,.15,len(f)),0,1)
            f["issue_time"]=issue; f["available_time"]=issue
            rows.append(f.reset_index())
    pd.concat(rows,ignore_index=True).to_csv(root/"forecasts.csv",index=False)
    write_json(root/"manifest.json",dict(mode="synthetic",availability_verified=False,
                                         note="All data simulated; no real weather or model-skill conclusion"))
    return root


def main():
    p=argparse.ArgumentParser(); p.add_argument("--directory",default="demo_run")
    p.add_argument("--generate-only",action="store_true")
    a=p.parse_args(); root=Path(a.directory).resolve()
    data=generate(root/"data")
    cfg=json.loads((Path(__file__).resolve().parents[1]/"config.json").read_text(encoding="utf-8"))
    cfg.update(data_dir=str(data),output_dir=str(root/"results"),mode="synthetic",site_id="DEMO",
               train_years=[2018,2019],val_years=[2020],test_years=[2021],decision_delay_hours=0,
               ablations=["base","season","weather","extreme"],seeds=[11,23],hidden=16,
               layers=1,heads=2,degree=8,epochs=2,patience=2,batch_size=16,bootstrap_repeats=100,
               device="cpu")
    write_json(root/"demo_config.json",cfg)
    if not a.generate_only:
        run(cfg,"validate"); run(cfg,"test")


if __name__=="__main__":
    main()
