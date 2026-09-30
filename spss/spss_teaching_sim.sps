* ================================================================.
* Simplified Monte Carlo simulation inside SPSS (MATRIX language).
* Teaching version: ONE error law, ONE sample size, FIXED bandwidth
* (no cross-validation), so numbers are NOT comparable with the paper's tables.
*
* CORRECTED VERSION - verified by running in IBM SPSS Statistics.
* The original file (written without access to SPSS) had two bugs,
* fixed here:
*   1) SET MXLOOP=100000.  added -- SPSS's default loop limit (40)
*      is far too small for this code's loops (101 and 200 iterations).
*   2) The function NORMAL(n,1) does not exist in the MATRIX language.
*      Normal errors are generated here with the Box-Muller transform
*      from two uniform random numbers instead.
*
* Reference values computed in Python for the same setting
* (Normal errors, n = 200, h = 0.15, grid of 101 points on [0.1,1.9]):
*   mean MSE ~ 0.022, median ~ 0.019, SD ~ 0.013, P90 ~ 0.038
* Actually obtained when run (R = 200 replications, same seed):
*   mean = 0.0211, median = 0.0185, SD = 0.01305 -- consistent with the above.
* Cauchy errors scaled to N(0,1) IQR (factor 0.6745), same n and h:
*   median ~ 2.4 ; mean is unstable (hundreds) - a symptom of infinite variance.
* Actually obtained when run (n = 500, R = 500): mean = 588.31, median = 2.2526.
* ================================================================.

SET SEED=20260918.
SET MXLOOP=100000.

MATRIX.
COMPUTE nrep = 200.
COMPUTE nobs = 200.
COMPUTE bw = 0.15.
COMPUTE ng = 101.

* Evaluation grid and true function.
COMPUTE xg = MAKE(ng,1,0).
LOOP j = 1 TO ng.
  COMPUTE xg(j,1) = 0.1 + 1.8*(j-1)/(ng-1).
END LOOP.
COMPUTE fg = SIN(xg) + 0.5*SIN(3*xg).

COMPUTE res = MAKE(nrep,1,0).
COMPUTE ones_n = MAKE(1,nobs,1).
COMPUTE ones_g = MAKE(ng,1,1).

LOOP k = 1 TO nrep.
  COMPUTE x = 2*UNIFORM(nobs,1).
  * --- error law: Normal (default), generated via Box-Muller from two
  *     uniform draws, since MATRIX has no built-in NORMAL() function.
  *     To use Cauchy instead, comment the three Normal lines below and
  *     uncomment the two Cauchy lines beneath them.
  COMPUTE u1 = UNIFORM(nobs,1).
  COMPUTE u2 = UNIFORM(nobs,1).
  COMPUTE eps = SQRT(-2*LN(u1)) &* COS(6.283185307179586*u2).
  * COMPUTE u = UNIFORM(nobs,1).
  * COMPUTE eps = 0.6745*( SIN(3.14159265358979*(u-0.5)) &/ COS(3.14159265358979*(u-0.5)) ).
  COMPUTE y = SIN(x) + 0.5*SIN(3*x) + eps.
  * --- Nadaraya-Watson estimate on the grid (Gaussian kernel, bandwidth bw).
  COMPUTE d = xg*ones_n - ones_g*T(x).
  COMPUTE w = EXP(-0.5*(((1/bw)*d)&**2)).
  COMPUTE fh = (w*y) &/ RSUM(w).
  COMPUTE res(k,1) = CSUM(RSUM((fh-fg)&**2))/ng.
END LOOP.

SAVE res /OUTFILE=* /VARIABLES=mse.
END MATRIX.

FREQUENCIES VARIABLES=mse
  /FORMAT=NOTABLE
  /STATISTICS=MEAN MEDIAN STDDEV MINIMUM MAXIMUM
  /PERCENTILES=25 75 90.

EXAMINE VARIABLES=mse
  /PLOT=BOXPLOT HISTOGRAM
  /STATISTICS=DESCRIPTIVES.

* Exercises: change bw (0.10, 0.25), nobs (100, 500, 1000), nrep (1000),
* and compare the mean, the median and the spread of MSE.
