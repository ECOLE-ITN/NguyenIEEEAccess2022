# Reproduce Second Experiment (Appendix B.2)
In general, to reproduce this experiments, you simply run a couple of scripts. 

## 1- Install relevant libraries :
```r
pip install -r installed_modules.txt
```
## 2- Download data to local
Hint: in our experience, OpenML might freeze when we download too many datasets at the same time. 
```r
sbatch loaddata.c
```
## 2- Run experiment:

```r
sbatch ExpRun.c
```

Once the optimization is completed, the final results will automatically save to `AutoMLExperiment/Automl_DACOpt.csv` and the corresponding logs at `AutoMLExperiment/logs`

