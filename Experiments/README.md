# Reproduce Experiments

## First Experiments (Appendix B.1)
Our productive scripts for the first experiment are provided in the sub-folder [FirstExperiment](FirstExperiment). Please read the respective [readme](FirstExperiment/README.md) instruction before running the experiment. In order to achieve the reproducibility of results of our first experiments on a Linux server, please use a similar architecture as listed below:

| Computing Infrastructure | Linux Cluster           |
| ------------------------ | ----------------------- |
| Architecture             | x86_64                  |
| O.S.                     | [Ubuntu Linux (20.04.3)](http://www.ubuntu.com/) |
| CPU(s)                   | 64             |
| Model name               | Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz|
| Speed                    | 2.1 GHz                 |


| Topic   | Description        | Link                                    |
| ------- | ------------------ | --------------------------------------- |
| Dataset | 44 Imbalanced data sets| [KEEL](https://sci2s.ugr.es/keel/imbalanced.php) |
| Enviroment| Library required| [link](FirstExperiment/installed_modules.txt)  |
|          | Python 3.7.2  | [link](https://www.python.org/downloads/release/python-372/)  |



## Second Experiments (Appendix B.2)

The productive scripts for the second experiment are provided in the sub-folder [SecondExperiment](SecondExperiment). Please read the respective [readme](SecondExperiment/README.md) instruction before running the experiment. Note that, the second experiment based on a submission to a Linux
cluster with [Slurm](https://slurm.schedmd.com/documentation.html) as workload manager. 
In order to achieve the reproducibility of results of our second experiments, please use a similar architecture as listed below:

| Computing Infrastructure | Linux Cluster           |
| ------------------------ | ----------------------- |
| Architecture             | x86_64                  |
| O.S.                     | [CentOS Linux 7](http://www.centos.org/)  |
| CPU                      | dual 8-core             |
| Model name               |Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz|
| Speed                    | 2.4 GHz                 |
| Cores per Node           | 32                      |
| Memory limit             | 64 GB                   |
| Network                  | IB and GbE              |

| Topic   | Description        | Link                                    |
| ------- | ------------------ | --------------------------------------- |
| Dataset | 73 Imbalanced data sets| [OpenML](https://openml.org) |
| Enviroment| Python 3.7.2  | [link](https://www.python.org/downloads/release/python-372/) |
|          | Library required| [link](SecondExperiment/installed_modules.txt) |
|          | Auto-sklearn| [installation notes](https://automl.github.io/auto-sklearn/master/installation.html)|
|          | Swig 3.0.10| [tutorial](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/) |



