import os, sys, time, numpy as np
from simlib import *
OUT="results"; os.makedirs(OUT,exist_ok=True)
SIZES=[50,100,200,500,1000,2000]; R=1000
t0=time.time()
for dist in DISTS:
    di=STREAM[dist]
    for n in SIZES:
        fn=f"{OUT}/{dist.replace('(','').replace(')','')}_{n}.npz"
        if os.path.exists(fn): continue
        rng=np.random.default_rng(np.random.SeedSequence([20260918,di,n]))
        robust = dist in SYMMETRIC
        keys=["h_nw","mse_nw","mse_nw_full"]+(["h_rb","mse_rb","mse_rb_full"] if robust else [])
        store={k:np.empty(R) for k in keys}
        fnw=np.empty((R,len(XG)),np.float32)
        frb=np.empty((R,len(XG)),np.float32) if robust else None
        for r in range(R):
            o=one_rep(dist,n,rng,robust)
            for k in keys: store[k][r]=o[k]
            fnw[r]=o["fh_nw"]
            if robust: frb[r]=o["fh_rb"]
        extra={"fh_nw":fnw}
        if robust: extra["fh_rb"]=frb
        np.savez_compressed(fn,**store,**extra)
        print(f"done {dist} n={n}  elapsed {time.time()-t0:.0f}s",flush=True)
print("ALL DONE",flush=True)
