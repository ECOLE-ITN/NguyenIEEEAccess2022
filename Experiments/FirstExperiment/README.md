# First Experiment (Appendix B.1)
Reproduce Experiment scripts
General speaking, to reproduce the first experiments, you simply download data from [KEEL](https://sci2s.ugr.es/keel/imbalanced.php) to the folder `DATA` run the script `imbalance2022.py`. 

## 1- Install relevant libraries :
```r
pip install -r installed_modules.txt
```

## 2- Run experiment:

```r
python imbalance2022.py [dataname] [seed] [underlying optimizer] [eliminate criteria]
```
### 2.1 An example run experiment on dataset `glass1`: 
2.1.1- Run DAC-HB
```r
python imbalance2022.py glass1 1 bo4ml highest
```

2.1.2- Run DAC-SB
```r
python imbalance2022.py glass1 1 bo4ml stat
```

2.1.3- Run DAC-HH
```r
python imbalance2022.py glass1 1 hpo highest
```

2.1.2- Run DAC-SH
```r
python imbalance2022.py glass1 1 hpo stat
```

Once the optimization is completed, the final results will automatically save to `RESULTS/DACOpt_imbalance.csv` and the corresponding logs at `RESULTS/LOGS`

