"""Validation-driven training and locked, separate held-out evaluation."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .data import read_tables, build_samples, feature_names, Scaler, make_scaler
from .models import TiDE, Transformer, GRUNet, CNNBiGRU, crps_loss, twcrps_loss, crps_loss_masked, add_te_head, te_quantiles, TE_SCALE
from .metrics import TAU, score, case_table, grouped, paired_bootstrap, event_diagnostics
from .postproc import emos_benchmark, conformal_shifts, apply_shifts


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def relative_paths(cfg):
    """Copy of cfg with every data/output path cut back to its part from processed/ or model_runs/, i.e. relative to
    the project root (10-28, release package)."""
    c=json.loads(json.dumps(cfg))
    def rel(p):
        p=str(p).replace("\\","/")
        for key in ("/processed/","/model_runs/"):
            if key in p:
                return p[p.index(key)+1:]
        return p
    for k in ("data_dir","output_dir"):
        if k in c:
            c[k]=rel(c[k])
    for s in c.get("sites",[])+(c.get("pretrain") or {}).get("sites",[]):
        s["data_dir"]=rel(s["data_dir"])
    return c


def portable_fingerprint(cfg):
    """Fingerprint that is the same for a frozen config on any machine, whatever the project root is."""
    return fingerprint(relative_paths(cfg))


def frozen_match(selection,cfg):
    """The machine that froze the run checks the exact hash; a copy elsewhere checks the portable one written by pack_release.py."""
    return selection["config_hash"]==fingerprint(cfg) or selection.get("config_hash_portable")==portable_fingerprint(cfg)


def tensors(samples,scaler,with_te=False):
    parts=[torch.from_numpy(np.stack([scaler.transform(s["past"]) for s in samples])),
           torch.from_numpy(np.stack([getattr(scaler,"transform_future",scaler.transform)(s["future"]) for s in samples])),
           torch.from_numpy(np.stack([s["y"] for s in samples]))]
    if with_te:
        parts.append(torch.from_numpy(np.stack([s["y_te"] if s.get("y_te") is not None else np.full_like(s["y"],np.nan)
                                                for s in samples])))
    return TensorDataset(*parts)


def predict(model,loader,device):
    model.eval()
    chunks=[]
    with torch.no_grad():
        for batch in loader:
            chunks.append(model(batch[0].to(device),batch[1].to(device)).cpu().numpy())
    return np.concatenate(chunks).reshape(-1,99)


def predict_te(model,loader,device):
    model.eval()
    chunks=[]
    with torch.no_grad():
        for batch in loader:
            model(batch[0].to(device),batch[1].to(device))
            chunks.append(te_quantiles(model).cpu().numpy())
    return np.concatenate(chunks).reshape(-1,99)


def tw_term(q,y,tau,tw):
    """Sum of weight x threshold-weighted CRPS; tw_loss is one {threshold_m, weight} or a list of them
    (several thresholds put more weight on the upper tail, Wessel et al. 2025)."""
    return sum(t["weight"]*twcrps_loss(q,y,tau,t["threshold_m"]) for t in (tw if isinstance(tw,list) else [tw]))


def make_model(name,n_features,history,horizon,cfg):
    model=MODELS[name](n_features,history,horizon,cfg)
    return add_te_head(model,cfg) if cfg.get("te_target") else model


MODELS={"tide":TiDE,"transformer":Transformer,"gru":GRUNet,"cnn_bigru":CNNBiGRU}


def model_names(cfg):
    """Backbones to train; default (no "models" key) keeps the original TiDE + Transformer pair."""
    names=cfg.get("models",["tide","transformer"])
    unknown=[n for n in names if n not in MODELS]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose from {list(MODELS)}")
    return names


def train_one(name,seed,samples,scaler,cfg,checkpoint,device,pretrain=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    if str(device).startswith("cuda"):
        # Fused flash / memory-efficient attention kernels are non-deterministic on GPU; the math kernel
        # keeps runs with the same seed exactly reproducible (slightly slower, still far faster than CPU).
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cudnn.benchmark=False
    te=cfg.get("te_target")
    # Energy-aware term (10-23): CRPS of Hs^2 (wave power ~ Hs^2 Te). Hs quantiles are non-negative and ordered, so
    # their squares are the quantiles of Hs^2 and the term is still a proper score; scaled to Hs-like magnitude.
    en=cfg.get("energy_loss")
    ds=tensors(samples["train"],scaler,bool(te))
    val=tensors(samples["val"],scaler,bool(te))
    loader=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    vl=DataLoader(val,batch_size=cfg["batch_size"])
    model=make_model(name,ds.tensors[0].shape[-1],ds.tensors[0].shape[1],ds.tensors[1].shape[1],cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=1e-4)
    best,stale,history=float("inf"),0,[]
    tw=cfg.get("tw_loss")
    import time
    if pretrain:
        # Stage 1: pretraining on proxy-truth samples (e.g. ERA5 before the buoys existed). No checkpoint and
        # no early stopping here; stage 2 below fine-tunes on the observed-truth samples with early stopping.
        pl=DataLoader(tensors(pretrain,scaler,bool(te)),batch_size=cfg["batch_size"],shuffle=True,
                      generator=torch.Generator().manual_seed(seed+1000))
        for epoch in range(cfg["pretrain"]["epochs"]):
            t_epoch=time.time(); model.train(); total,count=0.,0
            for batch in pl:
                p,f,y=(b.to(device) for b in batch[:3])
                optimizer.zero_grad()
                q_out=model(p,f)
                loss=crps_loss(q_out,y,model.head.tau)
                if tw:
                    loss=loss+tw_term(q_out,y,model.head.tau,tw)
                if en:
                    loss=loss+en.get("weight",1.0)*crps_loss(q_out**2/en.get("scale",2.0),y**2/en.get("scale",2.0),model.head.tau)
                if te:
                    loss=loss+te.get("weight",1.0)*crps_loss_masked(te_quantiles(model)/TE_SCALE,batch[3].to(device)/TE_SCALE,model.head.tau)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); total+=loss.item()*len(p); count+=len(p)
            print(f"  {name} seed{seed} pretrain {epoch+1}/{cfg['pretrain']['epochs']}: train CRPS {total/count:.4f}  ({time.time()-t_epoch:.0f} s)",flush=True)
    for epoch in range(cfg["epochs"]):
        t_epoch=time.time()
        model.train(); total,count=0.,0
        for batch in loader:
            p,f,y=(b.to(device) for b in batch[:3])
            optimizer.zero_grad()
            q_out=model(p,f)
            loss=crps_loss(q_out,y,model.head.tau)
            if tw:
                # Extreme-aware training: CRPS + weight x threshold-weighted CRPS (still a proper score).
                loss=loss+tw_term(q_out,y,model.head.tau,tw)
            if en:
                loss=loss+en.get("weight",1.0)*crps_loss(q_out**2/en.get("scale",2.0),y**2/en.get("scale",2.0),model.head.tau)
            if te:
                # Multi-task: CRPS of the energy period (scaled to Hs-like magnitude); missing Te truth is masked.
                loss=loss+te.get("weight",1.0)*crps_loss_masked(te_quantiles(model)/TE_SCALE,batch[3].to(device)/TE_SCALE,model.head.tau)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); total+=loss.item()*len(p); count+=len(p)
        q=predict(model,vl,device)
        metric=float(score(q,val.tensors[2].numpy().ravel()).mean())
        history.append(dict(epoch=epoch+1,train_crps=total/count,val_crps=metric))
        te_msg=""
        if te:
            yte=val.tensors[3].numpy().ravel(); ok=np.isfinite(yte)
            te_val=float(score(predict_te(model,vl,device)[ok],yte[ok]).mean()) if ok.any() else float("nan")
            history[-1]["val_te_crps"]=te_val
            te_msg=f", val Te CRPS {te_val:.3f} s"
        flag=" *best" if metric<best-1e-6 else ""
        print(f"  {name} seed{seed} epoch {epoch+1}/{cfg['epochs']}: train loss {total/count:.4f}, "
              f"val CRPS {metric:.4f}{te_msg}{flag}  ({time.time()-t_epoch:.0f} s)",flush=True)
        if metric<best-1e-6:
            best,stale=metric,0
            torch.save({"model":model.state_dict(),"scaler":scaler.state(),"config":cfg,
                        "n_features":ds.tensors[0].shape[-1],"history":ds.tensors[0].shape[1],
                        "horizon":ds.tensors[1].shape[1],"model_name":name,"seed":seed},checkpoint)
        else:
            stale+=1
        if stale>=cfg["patience"]:
            break
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(state["model"])
    q=predict(model,vl,device)
    if te:
        # Te quantiles of the selected (early-stopped on Hs) checkpoint, saved beside the Hs validation file.
        np.savez_compressed(Path(checkpoint).with_name(Path(checkpoint).stem+"_val_te.npz"),quantiles=predict_te(model,vl,device))
    return q,history,sum(p.numel() for p in model.parameters())


def climatology(sites,samples,target):
    """Monthly climatological quantiles from each site's own training years (sites = load_sites output)."""
    tables={}
    def table(train):
        return (np.quantile(train,TAU),{m:np.quantile(train.loc[train.index.month==m],TAU)
                                         for m in range(1,13) if (train.index.month==m).sum()>=20})
    pooled=[]
    for c,obs,_ in sites:
        train=obs.loc[obs.index.year.isin(c["train_years"]),target].dropna()
        pooled.append(train)
        if len(train)>=20:
            tables[str(c["site_id"])]=table(train)
    if not tables:
        raise ValueError("Too few training observations for climatology")
    for c,obs,_ in sites:
        # A held-out site (no training years, leave-one-site-out) uses the pooled climatology of the
        # training sites: its own observations must not be used.
        tables.setdefault(str(c["site_id"]),table(pd.concat(pooled)))
    out=[]
    for s in samples:
        for row in s["metadata"]:
            allq,by_month=tables[str(row["site_id"])]
            out.append(by_month.get(pd.Timestamp(row["valid_time"]).month,allq))
    return np.stack(out)


PROJECT=Path(__file__).resolve().parents[3]


def attach_te(obs,site_id,cfg,proxy=False):
    """Second target te_obs: buoy spectral Te (te_target.truth_dir, default processed/buoy_spectral/<site>_*.csv);
    sites without spectra keep NaN (masked in the loss). Pretraining (proxy=True) uses the reanalysis Te column."""
    if proxy:
        obs["te_obs"]=obs["te"]
        return obs
    d=Path(cfg["te_target"].get("truth_dir","processed/buoy_spectral"))
    d=d if d.is_absolute() else PROJECT/d
    files=sorted(d.glob(f"{site_id}_*.csv"))
    if files:
        b=pd.read_csv(files[0],usecols=["valid_time","te"])
        b.index=pd.to_datetime(b.valid_time,utc=True)
        obs["te_obs"]=b.te.reindex(obs.index).to_numpy()
    else:
        obs["te_obs"]=np.nan
    return obs


def te_report(root,label,keys,samples_val,samples_train,names):
    """Validation report for the period head: CRPS (s) of the seed-averaged Te quantiles vs raw GEFS Te (point)
    and a training-years climatology (day of year ±15 d), per site; sites without observed Te are skipped."""
    files=[root/f"{k}_val_te.npz" for k in keys]
    if not all(f.exists() for f in files):
        return
    q=np.mean([np.load(f)["quantiles"] for f in files],axis=0)
    meta=pd.DataFrame([m for s in samples_val for m in s["metadata"]])
    y=np.concatenate([s["y_te"] for s in samples_val])
    raw=np.concatenate([s["future"][:,names.index("te")] for s in samples_val])
    tr=pd.DataFrame([dict(site=m["site_id"],t=m["valid_time"]) for s in samples_train for m in s["metadata"]])
    tr["y"]=np.concatenate([s["y_te"] for s in samples_train]); tr=tr.dropna()
    tr["doy"]=pd.to_datetime(tr.t,utc=True).dt.dayofyear
    doy=pd.to_datetime(meta.valid_time,utc=True).dt.dayofyear.to_numpy()
    clim=np.full((len(meta),99),np.nan)
    for site,g in tr.groupby("site"):
        k=(meta.site_id.astype(str)==str(site)).to_numpy()
        table={d:np.quantile(g.y[np.minimum(abs(g.doy-d),366-abs(g.doy-d))<=15],TAU) for d in np.unique(doy[k])}
        clim[k]=np.stack([table[d] for d in doy[k]])
    lines=[f"# 周期 Te 订正（验证集，{label}，各种子平均）","","| 站点 | 时刻数 | 模型 CRPS (s) | GEFS 原始 Te 绝对误差 (s) | 气候态 CRPS (s) | 相对 GEFS | P10–P90 覆盖率 | GEFS 平均偏差 (s) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for site in sorted(meta.site_id.astype(str).unique()):
        k=(meta.site_id.astype(str)==site).to_numpy()&np.isfinite(y)&np.isfinite(raw)
        if not k.any() or not np.isfinite(clim[k]).all():
            continue
        cm=score(q[k],y[k]).mean(); cr=np.abs(raw[k]-y[k]).mean(); cc=score(clim[k],y[k]).mean()
        p10,p90=np.array([np.interp(.1,TAU,r) for r in q[k]]),np.array([np.interp(.9,TAU,r) for r in q[k]])
        cov=((y[k]>=p10)&(y[k]<=p90)).mean()
        lines.append(f"| {site} | {k.sum()} | {cm:.3f} | {cr:.3f} | {cc:.3f} | {cm/cr-1:+.1%} | {cov:.0%} | {(raw[k]-y[k]).mean():+.2f} |")
    (root/"validation").mkdir(exist_ok=True)
    text="\n".join(lines)
    (root/"validation"/f"te_report_{label}.md").write_text(text+"\n",encoding="utf-8")
    np.savez_compressed(root/f"{label}_val_te_mean.npz",quantiles=q)
    print(text,flush=True)


def load_sites(cfg,te_proxy=False):
    """One site (classic config) or several pooled sites (cfg["sites"]: list of {site_id, data_dir,
    train_years, val_years, test_years}); each site keeps its own data, years and availability rules."""
    if "sites" not in cfg:
        obs,fc,manifest=read_tables(cfg)
        if cfg.get("te_target"):
            obs=attach_te(obs,cfg["site_id"],cfg,te_proxy)
        return [(cfg,obs,fc)],manifest
    out,manifests=[],{}
    for spec in cfg["sites"]:
        c=dict(cfg); c.pop("sites"); c.update(spec)
        obs,fc,manifest=read_tables(c)
        if cfg.get("te_target"):
            obs=attach_te(obs,spec["site_id"],cfg,te_proxy)
        out.append((c,obs,fc)); manifests[str(spec["site_id"])]=manifest
    return out,manifests


def build_all(sites,cfg,names,site_ids=None):
    """Samples of all sites merged; pooled runs get a one-hot site identifier appended to every input step.
    site_ids fixes the one-hot layout (pretraining sites use the main run's layout; a main-run site without
    pretraining data simply keeps a zero column), so both sample sets have the same number of inputs."""
    merged={"train":[],"val":[],"test":[]}; skipped={}
    ids=[str(c["site_id"]) for c,_,_ in sites]
    layout=[str(x) for x in site_ids] if site_ids else ids
    if any(x not in layout for x in ids):
        raise ValueError(f"Sites {ids} not all in the one-hot layout {layout}")
    onehot_on=len(layout)>1 and cfg.get("site_onehot",True)    # off for leave-one-site-out (unseen site)
    for i,(c,obs,fc) in enumerate(sites):
        samples,sk=build_samples(obs,fc,c,names)
        skipped[ids[i]]=sk
        for split,lst in samples.items():
            for smp in lst:
                if onehot_on:
                    onehot=np.zeros(len(layout),dtype=np.float32); onehot[layout.index(ids[i])]=1
                    smp["past"]=np.hstack([smp["past"],np.tile(onehot,(len(smp["past"]),1))])
                    smp["future"]=np.hstack([smp["future"],np.tile(onehot,(len(smp["future"]),1))])
                if len(ids)>1:
                    for m in smp["metadata"]:
                        if not m["event_id"]:      # background / unknown blocks are site-specific
                            m["block_id"]=f"{ids[i]}_{m['block_id']}"
                merged[split].append(smp)
    if onehot_on:
        names=names+[f"site_{x}" for x in layout]
    return merged,skipped,names


def objective(table):
    # Declared in advance: equal weight on overall and extreme-weather mean CRPS.
    extreme=table.typhoon|table.cold_surge
    return float(.5*table.crps.mean()+.5*table.loc[extreme,"crps"].mean()) if extreme.any() else float(table.crps.mean())


def baseline_tables(samples,clim,cfg,train=None,names=None):
    out=[case_table(samples,clim,"climatology",cfg,clim)]
    if train and names and "hs_spread" in names:
        # Statistical benchmark: EMOS per site x lead day on the training split (truncated normal and log-normal).
        for kind,q in emos_benchmark(train,samples,names).items():
            out.append(case_table(samples,q,f"emos_{kind}",cfg,clim))
    persistence=np.concatenate([np.full(len(s["y"]),s["persistence"]) for s in samples])
    raw=np.concatenate([s["raw_mean"] for s in samples])
    out.append(case_table(samples,np.repeat(persistence[:,None],99,axis=1),"persistence_point",cfg,clim))
    out.append(case_table(samples,np.repeat(raw[:,None],99,axis=1),"raw_mean_point",cfg,clim))
    if all(s["raw_quantiles"] is not None for s in samples):
        q=np.concatenate([s["raw_quantiles"] for s in samples])
        if not np.isfinite(q).all() or (np.diff(q,axis=-1)<-1e-5).any():
            raise ValueError("Invalid raw ensemble quantiles")
        out.append(case_table(samples,q,"raw_ensemble",cfg,clim))
    return out


def write_reports(root,tables,cfg):
    allcases=pd.concat(tables,ignore_index=True)
    allcases.to_csv(root/"case_scores.csv",index=False)
    grouped(allcases,cfg).to_csv(root/"grouped_scores.csv",index=False)
    event_diagnostics(allcases,cfg).to_csv(root/"event_diagnostics.csv",index=False)
    comparisons=[]
    names=list(allcases.model.unique())
    for i,a in enumerate(names):
        for b in names[i+1:]:
            aa,bb=allcases[allcases.model==a],allcases[allcases.model==b]
            comparisons.append(dict(a=a,b=b,**paired_bootstrap(aa,bb,cfg["bootstrap_repeats"])))
    write_json(root/"paired_comparisons.json",comparisons)
    lines=["# 分组诊断报告", "", f"数据模式：`{cfg['mode']}`；目标：`{cfg['target']}`。", ""]
    if cfg["mode"]!="real":
        lines += ["当前为模拟、再分析代理或理想/伪预报试验，不能据此宣称实际极端天气预测问题已解决。", ""]
    lines += ["数值由当前评分表计算。模型选择仅在验证阶段完成，测试诊断不应反过来调参。",
              "", "| 模型 | 天气型 | CRPS | 80%覆盖率 | 中位数偏差 | 独立事件数 |", "|---|---|---:|---:|---:|---:|"]
    for (model,weather),g in allcases.groupby(["model","weather"]):
        if model not in {"tide","transformer","gru","cnn_bigru","fusion","gated"}:
            continue
        ne=g.loc[g.event_id.fillna("").astype(str).ne(""),"event_id"].nunique()
        lines.append(f"| {model} | {weather} | {g.crps.mean():.4f} | {g.covered80.mean():.1%} | {g.error50.mean():+.4f} | {ne} |")
    lines += ["", "## 需要关注的现象", ""]
    flags=[]
    for (model,weather),g in allcases.groupby(["model","weather"]):
        if model not in {"tide","transformer","gru","cnn_bigru"}:
            continue
        if g.covered80.mean()<.75:
            flags.append(f"- {model} / {weather}：80% 区间覆盖率为 {g.covered80.mean():.1%}，提示概率分布可能过窄或有系统偏差；先检查输入误差与校准。")
        if weather in {"typhoon","cold_surge","typhoon_and_cold"}:
            ne=g.loc[g.event_id.fillna("").astype(str).ne(""),"event_id"].nunique()
            if ne<cfg["min_group_events"]:
                flags.append(f"- {model} / {weather}：只有 {ne} 个具名独立事件，尚不足以支持稳定天气专用模型切换。")
            if g.error50.mean()<0:
                flags.append(f"- {model} / {weather}：中位数平均偏差 {g.error50.mean():+.4f}，检查极端低估、峰值时移、预报路径与强度偏差。")
    lines += flags or ["当前未触发预设提醒；仍需核查逐预见期评分和事件诊断，不能理解为不存在不足。"]
    lines += ["", "## 后续实验方向", "", "依据 validation/grouped_scores.csv 比较基础、季节、天气背景与极端特征的累进消融。",
              "极端时段输入误差明显时，优先补起报时可得的路径/强度不确定性、区域风压场和风浪/涌浪信息；若只是区间失准，再用独立校准子集检验概率校准。",
              "未实现自动因果归因或自动新增数据训练；任何新增方案需另建试验目录，保持测试集独立。"]
    (root/"diagnostic_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def gate_rules(a,b,cfg,fallback):
    rules={}
    for (regime,lead_band),g in a.groupby(["routing_regime","lead_band"]):
        other=b[(b.routing_regime==regime)&(b.lead_band==lead_band)]
        ci=paired_bootstrap(g,other,cfg["bootstrap_repeats"])
        event_count=g.loc[g.event_id.fillna("").astype(str).ne(""),"event_id"].nunique()
        enough=(event_count>=cfg["min_group_events"] if regime!="forecast_background_or_unknown" else ci["n_blocks"]>=cfg["min_group_events"])
        chosen=fallback
        if enough and ci["ci_high"] is not None:
            if ci["ci_high"]<0 and abs(ci["delta"])/max(other.crps.mean(),1e-8)>=.01:
                chosen="tide"
            elif ci["ci_low"]>0 and abs(ci["delta"])/max(g.crps.mean(),1e-8)>=.01:
                chosen="transformer"
        rules[f"{regime}|{lead_band}"]=dict(chosen=chosen,n_events=int(event_count),sufficient=bool(enough),**ci)
    return rules


def apply_gate(samples,predictions,rules,fallback):
    regimes=[f"{m['routing_regime']}|{m['lead_band']}" for s in samples for m in s["metadata"]]
    return np.stack([predictions[rules.get(r,{}).get("chosen",fallback)][i] for i,r in enumerate(regimes)])


def run(cfg,stage):
    root=Path(cfg["output_dir"]); root.mkdir(parents=True,exist_ok=True)
    if stage=="validate" and (root/"selection.json").exists():
        raise ValueError("Run directory already frozen; choose a new output_dir for another experiment")
    sites,manifest=load_sites(cfg)
    pre_sites=None
    if cfg.get("pretrain"):
        # Pretraining sites (proxy truth) in the same site order, so any site identifier lines up.
        pre_cfg=dict(cfg); pre_cfg["sites"]=cfg["pretrain"]["sites"]
        pre_sites,_=load_sites(pre_cfg,te_proxy=True)
    device=("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"]=="auto" else cfg["device"]
    torch.set_num_threads(min(torch.get_num_threads(),4))
    write_json(root/"config_used.json",cfg)
    if stage=="validate":
        seed_results=[]; all_tables=[]; summary={}; best={}; selected_pred={}; selected_samples=None
        models=model_names(cfg)
        n_runs=len(cfg["ablations"])*len(models)*len(cfg["seeds"]); run_no=0
        for ablation in cfg["ablations"]:
            names=feature_names(cfg,ablation)
            samples,skipped,names=build_all(sites,cfg,names)
            for split in ["train","val"]:
                if not samples[split]:
                    raise ValueError(f"No {split} samples. Check years, availability and sampling; skipped={skipped}")
            if cfg["mode"]!="synthetic" and len(samples["train"])<100:
                raise ValueError("Fewer than 100 training origins; not a research training set")
            scaler=make_scaler(cfg,names).fit(samples["train"])
            if hasattr(scaler,"components"):
                print(f"PCA: {len(scaler.cols)} inputs -> {scaler.components.shape[1]} components "
                      f"({sum(scaler.explained[:scaler.components.shape[1]]):.1%} of variance); model inputs {scaler.transform(samples['train'][0]['past']).shape[1]}",flush=True)
            pre_train=None
            if pre_sites:
                pre_samples,_,_=build_all(pre_sites,cfg,feature_names(cfg,ablation),
                                          site_ids=[c["site_id"] for c,_,_ in sites])
                pre_train=pre_samples["train"]
                print(f"Pretraining samples (proxy truth): {len(pre_train)}",flush=True)
            missing=[names[i] for i in scaler.all_missing]
            missing_rates={stream:dict(zip(names,np.mean(~np.isfinite(np.concatenate([s[stream] for s in samples["train"]])),axis=0).tolist()))
                           for stream in ["past","future"]}
            # Features declared past-only (e.g. analysed wind fields when no wind forecast exists) are
            # observed over the past window only; their future part is masked, which is allowed.
            past_only=set(cfg.get("past_only_features",[]))
            absent_future=[name for name,rate in missing_rates["future"].items() if rate==1 and name not in past_only]
            summary[ablation]=dict(n_samples={s:len(v) for s,v in samples.items()},skipped=skipped,
                                   features=names,all_missing_features=missing,missing_rates=missing_rates)
            if (missing or absent_future) and cfg["mode"]!="synthetic":
                raise ValueError(f"Ablation {ablation} has entirely missing input features: {missing}, future-only missing: {absent_future}; supply data or revise declared ablations")
            clim=climatology(sites,samples["val"],cfg["target"])
            if selected_samples is None:
                all_tables.extend(baseline_tables(samples["val"],clim,cfg,samples["train"],names))
            for model_name in models:
                qs=[]
                for seed in cfg["seeds"]:
                    key=f"{model_name}_{ablation}_seed{seed}"
                    checkpoint=root/f"{key}.pt"
                    done_files=[checkpoint,root/f"{key}_val.npz",root/f"{key}_history.json"]
                    if all(f.exists() for f in done_files):
                        # Resume after interruption: reuse a finished run (same config, deterministic samples).
                        q=np.load(root/f"{key}_val.npz")["quantiles"]
                        history=json.loads((root/f"{key}_history.json").read_text(encoding="utf-8"))
                        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
                        params=sum(p.numel() for p in make_model(model_name,state["n_features"],state["history"],
                                                                 state["horizon"],cfg).parameters())
                        run_no+=1
                        print(f"[{run_no}/{n_runs}] Resumed finished run {key} (no retraining)",flush=True)
                    else:
                        run_no+=1
                        print(f"[{run_no}/{n_runs}] Training {key} ({len(samples['train'])} train origins)",flush=True)
                        q,history,params=train_one(model_name,seed,samples,scaler,cfg,checkpoint,device,pre_train)
                    qs.append(q)
                    np.savez_compressed(root/f"{key}_val.npz",quantiles=q)
                    write_json(root/f"{key}_history.json",history)
                    seed_results.append(dict(model=model_name,ablation=ablation,seed=seed,parameters=params,
                                             val_crps=float(score(q,np.concatenate([s['y'] for s in samples['val']])).mean())))
                # This is a quantile average across seeds (Wasserstein barycenter), not a distribution mixture.
                q=np.mean(qs,axis=0)
                label=f"{model_name}_{ablation}"
                if cfg.get("te_target"):
                    te_report(root,label,[f"{model_name}_{ablation}_seed{sd}" for sd in cfg["seeds"]],
                              samples["val"],samples["train"],names)
                tab=case_table(samples["val"],q,label,cfg,clim)
                all_tables.append(tab)
                obj=objective(tab)
                if model_name not in best or obj<best[model_name]["objective"]:
                    best[model_name]=dict(ablation=ablation,objective=obj)
                    selected_pred[model_name]=q
            selected_samples=samples["val"]
        # All ablations share exactly the same cases; missing features are masked, not dropped.
        clim=climatology(sites,selected_samples,cfg["target"])
        pair="tide" in selected_pred and "transformer" in selected_pred
        if pair:
            # Explicit distribution mixture: interpolate its CDF on a dense quantile grid.
            selected_pred["fusion"]=mixture_quantiles(selected_pred["tide"],selected_pred["transformer"])
        tabs={m:case_table(selected_samples,q,m,cfg,clim) for m,q in selected_pred.items()}
        fallback=min(tabs,key=lambda m:objective(tabs[m]))
        if pair:
            rules=gate_rules(tabs["tide"],tabs["transformer"],cfg,fallback)
            gateq=apply_gate(selected_samples,selected_pred,rules,fallback)
            gatetab=case_table(selected_samples,gateq,"gated",cfg,clim)
            # Gate remains a candidate unless it improves the prespecified validation objective.
            deploy="gated" if objective(gatetab)<objective(tabs[fallback])*.99 else fallback
            all_tables+=list(tabs.values())+[gatetab]
        else:
            rules={}; deploy=fallback
            all_tables+=list(tabs.values())
        report=root/"validation"; report.mkdir(exist_ok=True)
        write_reports(report,all_tables,cfg)
        pd.DataFrame(seed_results).to_csv(root/"seed_results.csv",index=False)
        write_json(root/"data_audit.json",dict(mode=cfg["mode"],source_manifest=manifest,ablations=summary))
        selection=dict(config_hash=fingerprint(cfg),config_hash_portable=portable_fingerprint(cfg),best=best,fallback=fallback,rules=rules,deploy_candidate=deploy,
                       selection_split="validation_only",objective="0.5 overall CRPS + 0.5 extreme-weather CRPS; overall if no extremes",
                       warning="Validation scores used for selection are optimistic. No real-skill claim before held-out evaluation.")
        write_json(root/"selection.json",selection)
        print(f"Validation complete. Frozen candidate: {deploy}. Test set has not been scored.",flush=True)
    else:
        selection=json.loads((root/"selection.json").read_text(encoding="utf-8"))
        if not frozen_match(selection,cfg):
            raise ValueError("Config differs from frozen validation selection")
        predictions={}; samples_test=None
        for name in model_names(cfg):
            ablation=selection["best"][name]["ablation"]
            samples,_,names_test=build_all(sites,cfg,feature_names(cfg,ablation))
            if not samples["test"]:
                raise ValueError("No independent test origins available")
            qs,tes=[],[]
            for seed in cfg["seeds"]:
                state=torch.load(root/f"{name}_{ablation}_seed{seed}.pt",map_location=device,weights_only=False)
                scaler=Scaler.from_state(state["scaler"])
                model=make_model(name,state["n_features"],state["history"],state["horizon"],cfg).to(device)
                model.load_state_dict(state["model"])
                loader=DataLoader(tensors(samples["test"],scaler),batch_size=cfg["batch_size"])
                qs.append(predict(model,loader,device))
                if cfg.get("te_target"):
                    tes.append(predict_te(model,loader,device))
            predictions[name]=np.mean(qs,axis=0); samples_test=samples["test"]; samples_train=samples["train"]
            if tes:
                # Seed-averaged Te quantiles (s) of the period head; read by export_quantile_forecasts.py --te-config.
                (root/"test").mkdir(exist_ok=True)
                np.savez_compressed(root/"test"/"te_quantiles.npz",quantiles=np.mean(tes,axis=0))
            # Sensitivity variant: quantile-wise conformal shifts per site x lead band, fitted on the
            # seed-averaged validation predictions only (Romano et al. 2019); reported beside the frozen candidate.
            qval=np.mean([np.load(root/f"{name}_{ablation}_seed{sd}_val.npz")["quantiles"] for sd in cfg["seeds"]],axis=0)
            key=lambda smp:np.array([f"{m['site_id']}|{m['lead_band']}" for s in smp for m in s["metadata"]])
            shifts=conformal_shifts(qval,np.concatenate([s["y"] for s in samples["val"]]),key(samples["val"]))
            predictions[f"{name}_cqr"]=apply_shifts(predictions[name],key(samples_test),shifts)
        if "tide" in predictions and "transformer" in predictions:
            predictions["fusion"]=mixture_quantiles(predictions["tide"],predictions["transformer"])
            predictions["gated"]=apply_gate(samples_test,predictions,selection["rules"],selection["fallback"])
        clim=climatology(sites,samples_test,cfg["target"])
        tables=baseline_tables(samples_test,clim,cfg,samples_train,names_test)
        tables += [case_table(samples_test,q,name,cfg,clim) for name,q in predictions.items()]
        report=root/"test"; report.mkdir(exist_ok=True)
        write_reports(report,tables,cfg)
        np.savez_compressed(report/"quantiles.npz",**predictions)
        print(f"Held-out evaluation complete; frozen candidate={selection['deploy_candidate']}",flush=True)


def mixture_quantiles(a,b):
    out=[]
    for x,y in zip(a,b):
        grid=np.unique(np.concatenate([x,y,[0.]]))
        cdf=.5*np.interp(grid,x,TAU,left=0,right=1)+.5*np.interp(grid,y,TAU,left=0,right=1)
        out.append(np.interp(TAU,cdf,grid))
    return np.asarray(out,dtype=np.float32)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="config.json")
    parser.add_argument("--stage",choices=["validate","test"],default="validate")
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    run(cfg,args.stage)


if __name__=="__main__":
    main()
