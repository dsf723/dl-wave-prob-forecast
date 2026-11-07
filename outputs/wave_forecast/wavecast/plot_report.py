"""Export static scientific diagnosis plots from completed experiment reports."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory,title):
    directory=Path(directory)
    df=pd.read_csv(directory/"case_scores.csv")
    models=[m for m in ["tide","transformer","fusion","gated","raw_ensemble","climatology"] if m in set(df.model)]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for model in models:
        g=df[df.model==model]
        axes[0,0].plot(g.groupby("lead_day").crps.mean(),label=model,marker=".")
        axes[0,1].plot(g.groupby("lead_day").covered80.mean(),label=model,marker=".")
    axes[0,0].set(xlabel="Lead day",ylabel="Mean CRPS",title="Overall probability score (lower is better)")
    axes[0,0].legend(fontsize=8)
    axes[0,1].axhline(.8,color="black",linestyle="--",linewidth=1)
    axes[0,1].set(xlabel="Lead day",ylabel="80% interval coverage",ylim=(0,1),title="Calibration by lead time")
    selected=df[df.model.isin(models)]
    weather=selected.pivot_table(index="weather",columns="model",values="crps",aggfunc="mean").reindex(columns=models)
    weather.plot.bar(ax=axes[1,0],rot=15)
    axes[1,0].set(ylabel="Mean CRPS",title="Weather diagnosis; inspect independent event counts")
    axes[1,0].legend(fontsize=7)
    extreme=selected[selected.typhoon|selected.cold_surge]
    for model in models:
        g=extreme[extreme.model==model]
        axes[1,1].plot(g.groupby("lead_day").crps.mean(),label=model,marker=".")
    axes[1,1].set(xlabel="Lead day",ylabel="Mean CRPS",title="Extreme-weather cases only")
    fig.suptitle(title,fontsize=14)
    fig.savefig(directory/"diagnosis.png",dpi=160)
    fig.savefig(directory/"diagnosis.pdf")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("directory"); p.add_argument("--title",default="Wave probability forecast diagnosis")
    a=p.parse_args(); plot(a.directory,a.title)
