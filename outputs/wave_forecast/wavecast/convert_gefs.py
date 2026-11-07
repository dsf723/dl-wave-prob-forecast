"""Convert one existing member-point archive; availability is supplied, never guessed."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import TAU


def convert(path,issue_time,available_time,site_id,target="hs",period_verified=False,min_members=25):
    df=pd.read_csv(path)
    issue=pd.Timestamp(issue_time); available=pd.Timestamp(available_time)
    issue=issue.tz_localize("UTC") if issue.tz is None else issue.tz_convert("UTC")
    available=available.tz_localize("UTC") if available.tz is None else available.tz_convert("UTC")
    if available<issue:
        raise ValueError("Availability cannot precede initialization")
    if target=="te" and not period_verified:
        raise ValueError("Verify period definition before treating GEFS mwp as energy period Te")
    df["valid_time"]=pd.to_datetime(df.valid_time,utc=True,format="mixed")
    if df.duplicated(["member","valid_time"]).any():
        raise ValueError("Duplicate member-valid_time keys")
    df=df.rename(columns={"swh":"hs","u":"u10","v":"v10"})
    if period_verified and "mwp" in df:
        df=df.rename(columns={"mwp":"te"})
    start=df.valid_time.min(); end=df.valid_time.max()
    grid=pd.date_range(start,end,freq="3h")
    variables=[c for c in ["hs","te","u10","v10","mwd"] if c in df]
    members=[]
    for member,g in df.groupby("member"):
        g=g.set_index("valid_time").sort_index()
        out=pd.DataFrame(index=grid)
        for col in variables:
            s=g[col].dropna()
            if col=="mwd":
                parts=[np.sin(np.deg2rad(s)),np.cos(np.deg2rad(s))]
            else:
                parts=[s]
            values=[]
            for part in parts:
                union=part.index.union(grid)
                z=part.reindex(union).interpolate(method="time",limit_area="inside").reindex(grid)
                left=pd.Series(s.index,index=s.index).reindex(union).ffill().reindex(grid)
                right=pd.Series(s.index,index=s.index).reindex(union).bfill().reindex(grid)
                z[(right-left)>pd.Timedelta(hours=6)]=np.nan
                values.append(z)
            out[col]=(np.rad2deg(np.arctan2(values[0],values[1]))+360)%360 if col=="mwd" else values[0]
        out["member"]=member
        members.append(out.reset_index(names="valid_time"))
    complete=pd.concat(members,ignore_index=True)
    grouped=complete.groupby("valid_time")
    out=grouped[variables].mean()
    if "mwd" in variables:
        radians=np.deg2rad(complete.mwd)
        ss=np.sin(radians).groupby(complete.valid_time).mean()
        cc=np.cos(radians).groupby(complete.valid_time).mean()
        out["mwd"]=(np.rad2deg(np.arctan2(ss,cc))+360)%360
    out["hs_spread"]=grouped.hs.std(ddof=0)
    counts=grouped[target].count()
    quantiles=grouped[target].quantile(TAU).unstack()
    quantiles.columns=[f"q{i:03d}" for i in range(99)]
    out=out.join(quantiles)
    out=out.loc[counts>=min_members]
    out["n_members"]=counts
    out["site_id"]=site_id; out["issue_time"]=issue; out["available_time"]=available
    out["source"]="GEFS_member_archive"
    return out.reset_index()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--issue-time",required=True); p.add_argument("--available-time",required=True)
    p.add_argument("--site-id",required=True)
    p.add_argument("--target",choices=["hs","te"],default="hs")
    p.add_argument("--period-verified",action="store_true")
    p.add_argument("--min-members",type=int,default=25)
    a=p.parse_args()
    out=convert(a.input,a.issue_time,a.available_time,a.site_id,a.target,a.period_verified,a.min_members)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print(f"Converted {len(out)} forecast times. Publication timestamp must be backed by provenance.")


if __name__=="__main__":
    main()
