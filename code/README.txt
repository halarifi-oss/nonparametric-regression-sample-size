Monte Carlo study: error distribution and sample size in kernel regression
==========================================================================
Files
  simlib.py     Data generation, estimators (Nadaraya-Watson, robust weighted median),
                cross-validated bandwidth, one-replication function.
  run_sim.py    Runs all 5 error laws x 6 sample sizes x 1000 replications.
                Saves one .npz file per cell in a folder called results/ (resumable).
  analyze.py    Summary tables (summary_all.csv, slopes.csv).
  analyze2.py   Success rates, optimal n, figures (out/*.png).

How to run (Python 3.12, numpy 2.4, scipy 1.17, pandas, matplotlib)
  1. Put all four .py files in one folder.
  2. python run_sim.py      (about 30-60 minutes on one core)
  3. python analyze.py
  4. mkdir out ; python analyze2.py

Notes
  - Seeds are fixed (base seed 20260918 + fixed stream id per error law and sample size),
    so results are reproducible.
  - Errors are centred and rescaled to the IQR of N(0,1) (see simlib.py).
  - MSE is computed against the TRUE function on the interior region [0.1, 1.9].
