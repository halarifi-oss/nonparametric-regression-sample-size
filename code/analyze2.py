import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simlib import *
SIZES=np.array([50,100,200,500,1000,2000])
def load(d,n): return np.load(f"results/{d.replace('(','').replace(')','')}_{n}.npz")
df=pd.read_csv("summary_all.csv")
DELTAS=[0.02,0.01,0.005]
print("Var f(X) interior:",round(np.var(FG[INT]),4), " SD:",round(np.std(FG[INT]),3))

# success rate: share of replications with MSE <= delta
sr=[]
for d in DISTS:
    for n in SIZES:
        z=load(d,n)
        for est,tag in (("NW","nw"),("Robust","rb")):
            if tag=="rb" and d not in SYMMETRIC: continue
            m=z[f"mse_{tag}"]
            row=dict(dist=d,n=n,est=est)
            for dl in DELTAS: row[f"succ_{dl}"]=np.mean(m<=dl)
            row["frac_mse_gt_1"]=np.mean(m>1.0)
            sr.append(row)
sr=pd.DataFrame(sr); sr.to_csv("out/success_rates.csv",index=False)
print("=== success rate (MSE<=0.02), NW ===");print(sr[sr.est=="NW"].pivot(index="dist",columns="n",values="succ_0.02").reindex(DISTS).round(3))
print("=== success rate (MSE<=0.01), NW ===");print(sr[sr.est=="NW"].pivot(index="dist",columns="n",values="succ_0.01").reindex(DISTS).round(3))
print("=== success rate (MSE<=0.02), Robust ===");print(sr[sr.est=="Robust"].pivot(index="dist",columns="n",values="succ_0.02").round(3))
print("=== Cauchy NW share of reps with MSE>1 ===");print(sr[(sr.dist=="Cauchy")&(sr.est=="NW")][["n","frac_mse_gt_1"]].round(3).to_string(index=False))

# optimal n: (A) smallest n on grid with median MSE <= delta ; (B) smallest n with P90 <= delta (reliability)
def first_n(g,col,delta):
    ok=g[g[col]<=delta].sort_values("n")
    return int(ok.n.iloc[0]) if len(ok) else np.nan
res=[]
for (est,d),g in df.groupby(["est","dist"]):
    row=dict(est=est,dist=d)
    for dl in DELTAS:
        row[f"nA_{dl}"]=first_n(g,"median",dl); row[f"nB_{dl}"]=first_n(g,"p90",dl*4)
    # interpolated n* from log-log fit on n>=200 (asymptotic range)
    gg=g[g.n>=200]; b,a=np.polyfit(np.log(gg.n),np.log(gg["median"]),1)
    for dl in DELTAS:
        row[f"nfit_{dl}"]=np.exp((np.log(dl)-a)/b) if b<-0.3 else np.nan
    row["slope200+"]=b
    res.append(row)
res=pd.DataFrame(res); res.to_csv("out/optimal_n.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print("=== optimal n (A: median MSE<=delta | B: P90<=4*delta | fit) ===")
print(res.round(2).to_string(index=False))

# relative efficiency vs Normal (median MSE ratio) for NW
nw=df[df.est=="NW"].pivot(index="dist",columns="n",values="median")
print("=== ratio of median MSE to Normal (NW) ===");print((nw/nw.loc["Normal"]).round(2))
rel=df.pivot_table(index=["dist","n"],columns="est",values="median").dropna()
rel["rb/nw"]=rel["Robust"]/rel["NW"]; print("=== Robust/NW median MSE ratio ===");print(rel["rb/nw"].unstack().round(2))

# ----------------------------------------------------------- figures
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.3})
col={"Normal":"#1f77b4","Exponential":"#ff7f0e","Gamma":"#2ca02c","t(3)":"#9467bd","Cauchy":"#d62728"}
# Fig1 median MSE vs n
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
nwd=df[df.est=="NW"]
for d in DISTS:
    g=nwd[nwd.dist==d]; ax[0].plot(g.n,g["median"],"o-",c=col[d],label=d)
    ax[0].fill_between(g.n,g.q1,g.q3,color=col[d],alpha=.12)
ref=nwd[nwd.dist=="Normal"]; x=np.array([50,2000]); ax[0].plot(x,ref["median"].iloc[0]*(x/50)**-0.8,"k--",lw=1,label=r"$n^{-4/5}$ (theory)")
ax[0].set_xscale("log");ax[0].set_yscale("log");ax[0].set_xlabel("Sample size n");ax[0].set_ylabel("Median MSE (IQR band)")
ax[0].set_title("(a) Nadaraya-Watson kernel regression");ax[0].legend(fontsize=8)
rbd=df[df.est=="Robust"]
for d in ["Normal","t(3)","Cauchy"]:
    g=rbd[rbd.dist==d]; ax[1].plot(g.n,g["median"],"s-",c=col[d],label=d+" (robust)")
    g2=nwd[nwd.dist==d]; ax[1].plot(g2.n,g2["median"],"o:",c=col[d],alpha=.6,label=d+" (NW)")
ax[1].set_xscale("log");ax[1].set_yscale("log");ax[1].set_xlabel("Sample size n");ax[1].set_ylabel("Median MSE")
ax[1].set_title("(b) Robust weighted-median vs NW (symmetric errors)");ax[1].legend(fontsize=7,ncol=2)
plt.tight_layout();plt.savefig("out/fig1_median_mse_vs_n.png",dpi=200);plt.close()

# Fig2 boxplots of MSE by n (NW), 5 panels
fig,axs=plt.subplots(1,5,figsize=(15,3.8),sharey=True)
for a,d in zip(axs,DISTS):
    data=[np.log10(load(d,n)["mse_nw"]) for n in SIZES]
    bp=a.boxplot(data,labels=[str(n) for n in SIZES],showfliers=False,patch_artist=True)
    for b in bp["boxes"]: b.set_facecolor(col[d]);b.set_alpha(.5)
    a.set_title(d);a.set_xlabel("n")
axs[0].set_ylabel("log10(MSE)  [NW]")
plt.tight_layout();plt.savefig("out/fig2_boxplots_nw.png",dpi=200);plt.close()

# Fig3 fitted curves n=500
rng=np.random.default_rng(1)
fig,axs=plt.subplots(2,3,figsize=(12,6.5),sharex=True)
for j,d in enumerate(["Normal","t(3)","Cauchy"]):
    z=load(d,500)
    for i,(tag,nm) in enumerate((("nw","NW"),("rb","Robust"))):
        a=axs[i,j]; idx=rng.choice(1000,30,replace=False)
        for k in idx: a.plot(XG,z[f"fh_{tag}"][k],c="grey",alpha=.35,lw=.8)
        a.plot(XG,FG,c="red",lw=2); a.set_ylim(-1.5,2.5); a.set_title(f"{d} errors, n=500 - {nm}")
        if j==0: a.set_ylabel("estimate")
        if i==1: a.set_xlabel("x")
plt.tight_layout();plt.savefig("out/fig3_fitted_curves.png",dpi=200);plt.close()

# Fig4 Cauchy: mean vs median MSE and selected h
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
g=nwd[nwd.dist=="Cauchy"]
ax[0].plot(g.n,g["mean"],"o-",c="#d62728",label="mean MSE");ax[0].plot(g.n,g["median"],"s-",c="k",label="median MSE")
ax[0].set_xscale("log");ax[0].set_yscale("log");ax[0].set_xlabel("n");ax[0].set_title("NW under Cauchy errors");ax[0].legend()
ax[1].plot(g.n,g.h_med,"o-",c="#d62728");ax[1].axhline(0.9,ls="--",c="k",lw=1);ax[1].set_xscale("log");ax[1].set_xlabel("n");ax[1].set_ylabel("median selected h");ax[1].set_title("CV bandwidth (dashed = grid upper bound)")
plt.tight_layout();plt.savefig("out/fig4_cauchy.png",dpi=200);plt.close()
print("figures saved")
