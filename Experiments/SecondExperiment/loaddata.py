import json, logging, tempfile, sys, codecs, math, io, os,zipfile, time, copy, csv, arff
from ExpSupport import ExpSupport,OpenMLHoldoutDataManager 
task_id = int(sys.argv[1])
print(task_id)
OpenMLHoldoutDataManager(task_id).load(0.3)