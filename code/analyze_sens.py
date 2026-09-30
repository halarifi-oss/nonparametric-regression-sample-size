import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SIZES=[50,100,200,500,1000,2000]; LAWS=["Normal","t(3)","Cauchy"]
def key(l): return l.replace("(","").replace(")","")
S=pd.read_csv("summary_all.csv")
rows=[]
for c in ["noise0.5","noise2","func2"]:
    for l in LAWS:
        for n in SIZES:
            z=np.load(f"results_sens/{c}_{key(l)}_{n}.npz")
            for e,t in (("NW","nw"),("Robust","rb")):
                m=z[f"mse_{t}"]
                rows.append(dict(cond=c,dist=l,est=e,n=n,median=np.median(m),mean=m.mean(),p90=np.quantile(m,.9),succ=np.mean(m<=0.02),h_med=np.median(z[f"h_{t}"])))
for l in LAWS:
    for n in SIZES:
        for e in ("NW","Robust"):
            r=S[(S.dist==l)&(S.n==n)&(S.est==e)].iloc[0]
            rows.append(dict(cond="base",dist=l,est=e,n=n,median=r["median"],mean=r["mean"],p90=r["p90"],succ=np.nan,h_med=r["h_med"]))
D=pd.DataFrame(rows); D.to_csv("out/sensitivity_summary.csv",index=False)
pd.set_option("display.width",220)
for n in (500,2000):
    print(f"--- median MSE at n={n}")
    print(D[D.n==n].pivot_table(index=["dist","est"],columns="cond",values="median")[["base","noise0.5","noise2","func2"]].round(4))
print("--- ratios to base (median MSE): expected noise2 ~ 3.03, noise0.5 ~ 0.33 if MSE ~ sigma^1.6")
for n in (200,500,1000,2000):
    p=D[D.n==n].pivot_table(index=["dist","est"],columns="cond",values="median")
    print(f"n={n}"); print(pd.DataFrame({"x2":p["noise2"]/p["base"],"x0.5":p["noise0.5"]/p["base"],"f2":p["func2"]/p["base"]}).round(2))
# slopes n>=200
print("--- slopes (n>=200) log-log of median MSE")
for (c,l,e),g in D[D.n>=200].groupby(["cond","dist","est"]):
    b=np.polyfit(np.log(g.n),np.log(g["median"]),1)[0]; print(c,l,e,round(b,2))
# fitted n* for delta=0.02 (extrapolated if needed)
print("--- fitted n*(0.02), power law on n>=200")
out=[]
for (c,l,e),g in D[D.n>=200].groupby(["cond","dist","est"]):
    b,a=np.polyfit(np.log(g.n),np.log(g["median"]),1)
    nf=np.exp((np.log(0.02)-a)/b) if b<-0.3 else np.nan
    out.append(dict(cond=c,dist=l,est=e,slope=b,nfit=nf))
O=pd.DataFrame(out); O.to_csv("out/sensitivity_nfit.csv",index=False); print(O.pivot_table(index=["dist","est"],columns="cond",values="nfit").round(0)[["base","noise0.5","noise2","func2"]])

# figure
col={"base":"#1f77b4","noise0.5":"#2ca02c","noise2":"#d62728","func2":"#9467bd"}
lab={"base":"baseline","noise0.5":"noise x0.5","noise2":"noise x2","func2":"function 2"}
fig,ax=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
for a,l in zip(ax,LAWS):
    for c in ["base","noise0.5","noise2","func2"]:
        for e,ls in (("NW","-"),("Robust","--")):
            g=D[(D.cond==c)&(D.dist==l)&(D.est==e)]
            a.plot(g.n,g["median"],ls,marker="o" if e=="NW" else "s",ms=4,c=col[c],label=f"{lab[c]} - {e}")
    a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"{l} errors"); a.set_xlabel("Sample size n"); a.grid(alpha=.3)
ax[0].set_ylabel("Median MSE"); ax[0].legend(fontsize=6.5,ncol=2)
plt.tight_layout(); plt.savefig("/home/claude/paper/fig_sens.png",dpi=200,metadata={"Software":None})
print("saved")
