import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from simlib import *
from simlib import _spec
OUT="/home/claude/paper/"
S=pd.read_csv("summary_all.csv"); SR=pd.read_csv("out/success_rates.csv"); ON=pd.read_csv("out/optimal_n.csv")
SIZES=[50,100,200,500,1000,2000]
col={"Normal":"#1f77b4","Exponential":"#ff7f0e","Gamma":"#2ca02c","t(3)":"#9467bd","Cauchy":"#d62728"}
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.3})
meta={"Software":None}

# ---- design diagram
fig,ax=plt.subplots(figsize=(12.4,3.0)); ax.axis("off"); ax.set_xlim(0,12.4); ax.set_ylim(0,3)
labs=["1. Generate data\nX ~ U(0,2)\nY = f(X) + e","2. Error law e\n5 laws, centred,\nIQR-scaled","3. Estimate f\nNW kernel and\nrobust median\n(CV bandwidth)","4. Measure error\nMSE against the\ntrue f (interior)","5. Repeat 1000 times\n30 conditions\n(5 laws x 6 sizes)","6. Summarise\nmedian, IQR, P90,\nsuccess rate, n*"]
w=1.6; gap=0.48
for i,t in enumerate(labs):
    x=0.15+i*(w+gap)
    ax.add_patch(FancyBboxPatch((x,0.5),w,2.0,boxstyle="round,pad=0.03",fc="#EAF0FA",ec="#1F3864",lw=1.5))
    ax.text(x+w/2,1.5,t,ha="center",va="center",fontsize=9)
    if i<5: ax.add_patch(FancyArrowPatch((x+w+0.05,1.5),(x+w+gap-0.05,1.5),arrowstyle="-|>",mutation_scale=14,color="#1F3864",lw=1.5))
plt.tight_layout(); plt.savefig(OUT+"fig_design.png",dpi=200,metadata=meta); plt.close()

# ---- error densities (after centring and IQR scaling)
def pdf_scaled(name,x):
    d,mean,s=_spec(name); return d.pdf(x/s+mean)/s
fig,ax=plt.subplots(1,2,figsize=(11,4))
x=np.linspace(-6,6,1200); xl=np.linspace(-20,20,4000)
for d in DISTS:
    ax[0].plot(x,pdf_scaled(d,x),c=col[d],lw=2,label=d); ax[1].plot(xl,pdf_scaled(d,xl),c=col[d],lw=2,label=d)
ax[0].set_xlim(-6,6); ax[0].set_ylim(0,.75); ax[0].set_title("(a) Densities after centring and IQR scaling"); ax[0].set_xlabel("error e"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
ax[1].set_yscale("log"); ax[1].set_ylim(1e-6,1); ax[1].set_title("(b) Same densities, log scale (tails)"); ax[1].set_xlabel("error e")
plt.tight_layout(); plt.savefig(OUT+"fig_densities.png",dpi=200,metadata=meta); plt.close()

# ---- success rates
fig,ax=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for d in DISTS:
    g=SR[(SR.dist==d)&(SR.est=="NW")]; ax[0].plot(g.n,g["succ_0.02"],"o-",c=col[d],label=d)
for d in ["Normal","t(3)","Cauchy"]:
    g=SR[(SR.dist==d)&(SR.est=="Robust")]; ax[1].plot(g.n,g["succ_0.02"],"s-",c=col[d],label=d+" (robust)")
    g2=SR[(SR.dist==d)&(SR.est=="NW")]; ax[1].plot(g2.n,g2["succ_0.02"],"o:",c=col[d],alpha=.55,label=d+" (NW)")
for a,t in zip(ax,["(a) Kernel (NW) estimator","(b) Robust median vs NW (symmetric laws)"]):
    a.set_xscale("log"); a.set_xlabel("Sample size n"); a.set_title(t); a.axhline(.8,c="k",ls="--",lw=1)
ax[0].set_ylabel("Share of replications with MSE <= 0.02"); ax[0].legend(fontsize=8); ax[1].legend(fontsize=7,ncol=2,loc="lower right")
plt.tight_layout(); plt.savefig(OUT+"fig_success.png",dpi=200,metadata=meta); plt.close()

# ---- bias-variance
fig,ax=plt.subplots(figsize=(6.6,4.4))
for d in ["Normal","Exponential","Gamma","t(3)"]:
    g=S[(S.dist==d)&(S.est=="NW")]; ax.plot(g.n,g.ivar,"o-",c=col[d],label=d+" - variance"); ax.plot(g.n,g.ibias2,"s--",c=col[d],alpha=.7,label=d+" - squared bias")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Sample size n"); ax.set_ylabel("Integrated component of MSE"); ax.legend(fontsize=6.5,ncol=2)
ax.set_title("Variance (solid) vs squared bias (dashed), NW")
plt.tight_layout(); plt.savefig(OUT+"fig_biasvar.png",dpi=200,metadata=meta); plt.close()

# ---- optimal n bars
cats=[("Normal","NW"),("Exponential","NW"),("Gamma","NW"),("t(3)","NW"),("Cauchy","NW"),("Normal","Robust"),("t(3)","Robust"),("Cauchy","Robust")]
fig,ax=plt.subplots(figsize=(10,4.3)); wd=.26
for k,(dl,c) in enumerate([("nA_0.02","#4C78A8"),("nA_0.01","#F58518"),("nA_0.005","#54A24B")]):
    for i,(d,e) in enumerate(cats):
        v=ON[(ON.dist==d)&(ON.est==e)][dl].iloc[0]; xx=i+(k-1)*wd
        if np.isnan(v): ax.bar(xx,4000,width=wd,color="none",edgecolor=c,hatch="///",lw=1)
        else: ax.bar(xx,v,width=wd,color=c,label=("delta = "+dl.split("_")[1]) if i==0 else None)
ax.set_yscale("log"); ax.set_ylim(30,6000); ax.set_xticks(range(len(cats))); ax.set_xticklabels([f"{d}\n{'NW' if e=='NW' else 'Robust'}" for d,e in cats],fontsize=8)
ax.set_ylabel("Optimal sample size n*(delta)"); ax.legend(fontsize=8,loc="upper left"); ax.set_title("Optimal sample size by error law and estimator (hatched = not reached up to n = 2000)")
plt.tight_layout(); plt.savefig(OUT+"fig_optimaln.png",dpi=200,metadata=meta); plt.close()

# ---- strip metadata from existing figures
from PIL import Image
for f in ["fig1_median_mse_vs_n","fig2_boxplots_nw","fig3_fitted_curves","fig4_cauchy"]:
    im=Image.open(f"/home/claude/paper/{f}.png"); im.save(f"/home/claude/paper/{f}.png")
    print(f,im.size)
for f in ["fig_design","fig_densities","fig_success","fig_biasvar","fig_optimaln"]:
    print(f,Image.open(OUT+f+".png").size)

# ---- merged fig3+4: median MSE vs n (left,right) + boxplots row below
fig = plt.figure(figsize=(12,7.6))
gs = fig.add_gridspec(2,2, height_ratios=[1,1], hspace=0.38, wspace=0.22)
ax_a = fig.add_subplot(gs[0,0]); ax_b = fig.add_subplot(gs[0,1])
for d in DISTS:
    g=S[(S.dist==d)&(S.est=="NW")]; ax_a.plot(g.n,g["median"],"o-",c=col[d],label=d)
    ax_a.fill_between(g.n,g.q1,g.q3,color=col[d],alpha=.12)
ref=S[(S.dist=="Normal")&(S.est=="NW")]; xx=np.array([50,2000]); ax_a.plot(xx,ref["median"].iloc[0]*(xx/50)**-0.8,"k--",lw=1,label=r"$n^{-4/5}$ (theory)")
ax_a.set_xscale("log"); ax_a.set_yscale("log"); ax_a.set_xlabel("Sample size n"); ax_a.set_ylabel("Median MSE (IQR band)")
ax_a.set_title("(a) Nadaraya-Watson kernel regression"); ax_a.legend(fontsize=7.5)
for d in ["Normal","t(3)","Cauchy"]:
    g=S[(S.dist==d)&(S.est=="Robust")]; ax_b.plot(g.n,g["median"],"s-",c=col[d],label=d+" (robust)")
    g2=S[(S.dist==d)&(S.est=="NW")]; ax_b.plot(g2.n,g2["median"],"o:",c=col[d],alpha=.6,label=d+" (NW)")
ax_b.set_xscale("log"); ax_b.set_yscale("log"); ax_b.set_xlabel("Sample size n"); ax_b.set_ylabel("Median MSE")
ax_b.set_title("(b) Robust weighted-median vs NW (symmetric errors)"); ax_b.legend(fontsize=6.5,ncol=2)
gs2 = gs[1,:].subgridspec(1,5,wspace=0.08)
axs = [fig.add_subplot(gs2[0,i]) for i in range(5)]
for a,d in zip(axs,DISTS):
    data=[np.log10(np.load(f"results/{d.replace('(','').replace(')','')}_{n}.npz")["mse_nw"]) for n in SIZES]
    bp=a.boxplot(data,tick_labels=[str(n) for n in SIZES],showfliers=False,patch_artist=True)
    for b in bp["boxes"]: b.set_facecolor(col[d]); b.set_alpha(.5)
    a.set_title(d,fontsize=9); a.set_xlabel("n",fontsize=8); a.tick_params(labelsize=7)
axs[0].set_ylabel("log10(MSE)  [NW]",fontsize=8)
fig.text(0.02,0.47,"(c) Full distribution of MSE across replications (Nadaraya-Watson)",fontsize=9,ha="left")
plt.savefig(OUT+"fig_mse_combined.png",dpi=200,metadata=meta); plt.close()
from PIL import Image
print("fig_mse_combined",Image.open(OUT+"fig_mse_combined.png").size)
