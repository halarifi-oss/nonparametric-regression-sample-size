"""Sensitivity analysis: noise scale (x0.5, x2) and a second, more oscillatory regression function."""
import os, time, numpy as np
from simlib import *
OUT="results_sens"; os.makedirs(OUT,exist_ok=True)
SIZES=[50,100,200,500,1000,2000]; R=500
f2=lambda x: np.sin(2*x)+0.5*np.sin(6*x)
CONDS={"noise0.5":(f,0.5),"noise2":(f,2.0),"func2":(f2,1.0)}
LAWS=["Normal","t(3)","Cauchy"]
def one(dist,n,rng,fun,mult):
    X=rng.uniform(0,2,n); Y=fun(X)+mult*draw_error(dist,n,rng); F=fun(XG)
    h1=nw_cv_bandwidth(X,Y,rng); f1=nw_predict(XG,X,Y,h1)
    h2=wmedian_cv_bandwidth(X,Y,rng); f2_=wmedian_predict(XG,X,Y,h2)
    return h1,np.mean((f1[INT]-F[INT])**2),h2,np.mean((f2_[INT]-F[INT])**2)
t0=time.time()
for ci,(cname,(fun,mult)) in enumerate(CONDS.items()):
    for law in LAWS:
        for n in SIZES:
            fn=f"{OUT}/{cname}_{law.replace('(','').replace(')','')}_{n}.npz"
            if os.path.exists(fn): continue
            rng=np.random.default_rng(np.random.SeedSequence([20260918,100+ci,STREAM[law],n]))
            a=np.array([one(law,n,rng,fun,mult) for _ in range(R)])
            np.savez_compressed(fn,h_nw=a[:,0],mse_nw=a[:,1],h_rb=a[:,2],mse_rb=a[:,3])
            print(f"done {cname} {law} n={n} elapsed {time.time()-t0:.0f}s",flush=True)
print("ALL DONE",flush=True)
