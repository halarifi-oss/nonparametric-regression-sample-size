import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simlib import *
SIZES=[50,100,200,500,1000,2000]
def load(d,n):
    return np.load(f"results/{d.replace('(','').replace(')','')}_{n}.npz")
rows=[]
for d in DISTS:
    for n in SIZES:
        z=load(d,n)
        for est,tag in (("NW","nw"),("Robust","rb")):
            if tag=="rb" and d not in SYMMETRIC: continue
            m=z[f"mse_{tag}"]; fh=z[f"fh_{tag}"][:,INT].astype(float)
            ib2=np.mean((fh.mean(0)-FG[INT])**2); iv=np.mean(fh.var(0,ddof=1))
            rows.append(dict(dist=d,n=n,est=est,mean=m.mean(),mcse=m.std(ddof=1)/np.sqrt(len(m)),
                sd=m.std(ddof=1),median=np.median(m),q1=np.quantile(m,.25),q3=np.quantile(m,.75),
                p90=np.quantile(m,.90),ibias2=ib2,ivar=iv,h_med=np.median(z[f"h_{tag}"]),
                med_bias_abs=np.mean(np.abs(np.median(fh,0)-FG[INT]))))
df=pd.DataFrame(rows); df.to_csv("summary_all.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
nw=df[df.est=="NW"]
print("=== NW: median MSE ===");print(nw.pivot(index="dist",columns="n",values="median").reindex(DISTS).round(4))
print("=== NW: mean MSE ===");print(nw.pivot(index="dist",columns="n",values="mean").reindex(DISTS).round(4))
print("=== NW: P90 MSE ===");print(nw.pivot(index="dist",columns="n",values="p90").reindex(DISTS).round(4))
rb=df[df.est=="Robust"]
print("=== Robust: median MSE ===");print(rb.pivot(index="dist",columns="n",values="median").reindex([d for d in DISTS if d in SYMMETRIC]).round(4))
print("=== Robust: P90 ===");print(rb.pivot(index="dist",columns="n",values="p90").reindex([d for d in DISTS if d in SYMMETRIC]).round(4))
print("=== NW median h ===");print(nw.pivot(index="dist",columns="n",values="h_med").reindex(DISTS).round(3))
print("=== NW IBias2 ===");print(nw.pivot(index="dist",columns="n",values="ibias2").reindex(DISTS).round(4))
print("=== NW IVar ===");print(nw.pivot(index="dist",columns="n",values="ivar").reindex(DISTS).round(4))
# convergence-rate slopes (log median MSE ~ log n)
sl=[]
for est,g in df.groupby("est"):
    for d,gg in g.groupby("dist"):
        b=np.polyfit(np.log(gg.n),np.log(gg["median"]),1)[0]; sl.append((est,d,round(b,3)))
print("=== slopes (median MSE ~ n^slope; theory for NW = -0.8) ===")
for s in sl: print(s)
pd.DataFrame(sl,columns=["est","dist","slope"]).to_csv("slopes.csv",index=False)
