* ================================================================.
* Analysis of the Monte Carlo results in SPSS.
* NOTE: written without access to SPSS, so it is UNTESTED.
* Run block by block and compare with the Python summary tables.
* Edit the path in GET DATA to where mse_replications.csv is saved.
* ================================================================.

GET DATA
  /TYPE=TXT
  /FILE='C:\mc\mse_replications.csv'
  /ENCODING='UTF8'
  /DELIMITERS=","
  /QUALIFIER='"'
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /VARIABLES=
    dist A11
    n F5.0
    rep F5.0
    est A6
    mse F20.8
    log10mse F12.8
    h F8.5.
CACHE.
EXECUTE.
DATASET NAME mc WINDOW=FRONT.

COMPUTE log10n = LG10(n).
COMPUTE f_nw = (est = "NW").
COMPUTE f_nw4 = (est = "NW" AND dist NE "Cauchy").
EXECUTE.

* ---- 1. Summary table per distribution x sample size (NW estimator).
FILTER BY f_nw.
DATASET DECLARE summ.
AGGREGATE
  /OUTFILE='summ'
  /BREAK=dist n
  /N=N
  /mean_mse=MEAN(mse)
  /median_mse=MEDIAN(mse)
  /sd_mse=SD(mse)
  /succ_002=PLT(mse, 0.02).
DATASET ACTIVATE summ.
COMPUTE mcse = sd_mse / SQRT(N).
LIST.
DATASET ACTIVATE mc.

* ---- 2. Two-way ANOVA on log10(MSE), all five error laws.
* Exploratory only: variances are very unequal (Cauchy).
FILTER BY f_nw.
UNIANOVA log10mse BY dist n
  /METHOD=SSTYPE(3)
  /INTERCEPT=INCLUDE
  /EMMEANS=TABLES(dist*n)
  /PRINT=DESCRIPTIVE ETASQ HOMOGENEITY
  /DESIGN=dist n dist*n.

* ---- 3. Same analysis without Cauchy.
FILTER BY f_nw4.
UNIANOVA log10mse BY dist n
  /METHOD=SSTYPE(3)
  /INTERCEPT=INCLUDE
  /PRINT=DESCRIPTIVE ETASQ HOMOGENEITY
  /DESIGN=dist n dist*n.

* ---- 4. Log-log slope of MSE against n, per distribution.
* The slope is for the geometric mean of MSE, so it differs slightly
* from the median-based slopes in the paper (theory for NW: -0.8).
FILTER BY f_nw.
SORT CASES BY dist.
SPLIT FILE LAYERED BY dist.
REGRESSION
  /STATISTICS COEFF CI(95) R ANOVA
  /DEPENDENT log10mse
  /METHOD=ENTER log10n.
SPLIT FILE OFF.

* ---- 5. Kruskal-Wallis across error laws at n = 500.
AUTORECODE VARIABLES=dist /INTO dist_id /PRINT.
TEMPORARY.
SELECT IF (n = 500 AND est = "NW").
NPAR TESTS /K-W=mse BY dist_id(1,5).

* ---- 6. Median MSE by n (edit the chart to a log scale for the y-axis).
FILTER BY f_nw.
GRAPH /LINE(MULTIPLE)=MEDIAN(mse) BY n BY dist.

FILTER OFF.
