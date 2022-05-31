#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
import argparse, pickle
from autosklearn.data.xy_data_manager import XYDataManager
import numpy as np
import pandas as pd
import time,os,csv
from autosklearn.pipeline.classification import SimpleClassificationPipeline
#from autosklearn.pipeline import util as putil
from sklearn.metrics import accuracy_score
import scipy.sparse
from typing import Generator, Optional
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from hyperopt import STATUS_OK, STATUS_FAIL
import autosklearn.constants as ASConstants
from dacopt import DACOpt, ConfigSpace, ConditionalSpace, AlgorithmChoice, IntegerParam, FloatParam, CategoricalParam
from ExpSupport import ExpSupport,OpenMLHoldoutDataManager 
import json, logging, tempfile, sys, codecs, math, io, os,zipfile, time, copy, csv, arff
from ctypes import c_wchar_p 
from functools import partial
from hyperopt import STATUS_OK, Trials, STATUS_FAIL
from joblib.externals.loky.backend.context import get_context


# In[2]:


os.system("taskset -p 0xff %d" % os.getpid())
#https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy

# In[3]:


class RunBenchmark(object):
    errorcount=0
    successcount=0
    invalidcount=0
    timeoutcount=0
    xopt=None
    fopt=None
    def __init__(self,task_id, random_state,_slurmlog,_max_threads,_max_eval,number_candidates,_init_sample,
                 isDaC= True, eta =3,
                 compare_strategy='highest', suggest='tpe',BO_optimizer='bo4ml',
                 time_out= 3600,
                 _sample_sp= None, 
                 test_size= 0.3, n_folds= 4, load = True,
                ifAllSolution= False,show_message=False): 
        self.DATA_FOLDER='./AutoMLExperiment/logs'
        self.HOME_FOLDER='./AutoMLExperiment'
        self.task_id=task_id
        self.seed=random_state
        self.isDaC=isDaC
        self.eta=eta
        self.compare_strategy=compare_strategy
        self.show_message=show_message        
        self.suggest=suggest
        self.BO_optimizer=BO_optimizer
        self._slurmlog=_slurmlog       
        self._max_threads=_max_threads        
        self._max_eval=_max_eval
        self.number_candidates=number_candidates
        self._init_sample=_init_sample
        self._sample_sp=_sample_sp
        self.time_out=time_out
        self.n_EI_candidates=24#50 if self.BO_type in ('Bandit','full') else 24
        data = OpenMLHoldoutDataManager(task_id).load(test_size)
        self.dataset_id=data.dataset_id
        X_train, y_train, X_test, y_test = fold = data.folds[0]
        self.waittime=round(min(max(X_train.shape[0]/1000,0.01),10),2)
        self.categorical = data.categorical
        self.column_names = data.names        
        self.isSpare=data.isSpare
        self.multiclass=False
        if data.num_classes == 2:
            self.task = ASConstants.BINARY_CLASSIFICATION
        elif data.num_classes > 2:
            self.task = ASConstants.MULTICLASS_CLASSIFICATION
            self.multiclass=True
        else:
            raise ValueError('This benchmark needs at least two classes.')
        include, exclude={},{}
        self.data_manager = XYDataManager(X=X_train, y=y_train,X_test=X_test,y_test=y_test, 
                                          task=self.task, feat_type=data.variable_types,dataset_name=self.dataset_id)
        BO_type ='full' if self.isDaC or BO_optimizer.lower()=='bo4ml' else 'hpo'
        self.BO_type=BO_type
        self.support=ExpSupport(self.dataset_id,include, exclude,BO_type,self.task,self.isSpare)
        RunBenchmark.errorcount=0
        RunBenchmark.successcount=0
        RunBenchmark.invalidcount=0
        RunBenchmark.timeoutcount=0
        RunBenchmark.xopt=None
        RunBenchmark.fopt=1
    def _fit_and_score(self,config, X_train, y_train, X_test,y_test, score, e):        
        try:
            clf= SimpleClassificationPipeline(config, random_state =self.seed, dataset_properties= {'sparse': self.isSpare,'multiclass':self.multiclass}) 
            _predicted = clf.fit(X_train, y_train).predict(X_test)
            foldscore = accuracy_score(y_test, _predicted)
            score.value=foldscore
        except:
            e.value=1
    def _fake_fit_score(self,config, X_train, y_train, X_test,y_test, score, e):
        _return = np.random.uniform(0, 1)
        score.value= _return
        #print(_return)
        return _return
    def test_func(self,csparams):        
        score=0
        try:
            clf= SimpleClassificationPipeline(csparams, random_state =self.seed, dataset_properties= {'sparse': self.isSpare,'multiclass':self.multiclass})              
            _predicted = clf.fit(self.data_manager.data['X_train'], self.data_manager.data['Y_train']).predict(self.data_manager.data['X_test'])
            score = accuracy_score(self.data_manager.data['Y_test'], _predicted)
        except: 
            print('ERROR: test')
        return -score
    def fake_obj(self, csparams):
        #print('.')
        print(csparams)
        #print(params['p2']['name'],params['p7']['name'],params['p6']['name'],params['p3'],params['p4'],params['p5'])
        _return=np.random.uniform(0, 1)
        return {'loss': _return, 'status': STATUS_OK, 'runtime':0, 'msg':None}
    def objective_func(self,csparams):
        #print(csparams)
        start=time.time()
        trail_timeout=600
        ctx = mp.get_context('loky')
        manager = ctx.Manager()
        avg_score=[]
        exceptions=''
        score = manager.Value('d', 0)
        exception = manager.Value('i' , 0)
        self.i=self.i+1
        #csparams=self.support._getparamflat(csparams) 
        csparams=self.support.convertparams(csparams)
        kf = KFold(n_splits=4)
        #kf = StratifiedKFold(n_splits=4)
        X=self.data_manager.data['X_train']
        y=self.data_manager.data['Y_train']
        isTimeout=False
        fold_id=0
        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            p = ctx.Process(target=self._fit_and_score,
                                        args=(csparams, X_train, y_train, X_valid , y_valid , score,exception))
            p.start()
            #trail_timeout=trail_timeout - (time.time()-start)
            #timeout_fold=round(trail_timeout/(4-fold_id),1)
            p.join(150)# 150 seconds for 1/4 fold
            fold_id+=1
            if p.is_alive():
                isTimeout=True
                #avg_score.append(0)
                exceptions='timeout'
                p.terminate()
                p.join()
                #avg_score.append(score.value)
                break
            #print(score.value)
            avg_score.append(score.value)
            exceptions=exceptions+'-'+str(exception.value) 
            if exception.value==1:
                RunBenchmark.errorcount+=1
                break
           
        #avg_score.append(np.random.uniform(0, 1))
        loss = - np.mean(avg_score) if len(avg_score)>0 else 0       
        exceptions=avg_score if len(avg_score)>0 else 0
        if isTimeout:
            RunBenchmark.timeoutcount+=1
        if loss<0:
            RunBenchmark.successcount+=1
            if loss< RunBenchmark.fopt:
                RunBenchmark.fopt=loss
                RunBenchmark.xopt= csparams
                #print('***',self.i,loss,exceptions,time.time()-start)
        #print(self.i,loss,exceptions,time.time()-start)
        return {'loss': loss, 'status': STATUS_OK, 'runtime':time.time()-start, 'msg':exceptions}
    def logging(self,log_file_name, trials):
    #====Save details===        
        try:
            if self.BO_type in ('full','Bandit'):
                my_df = pd.DataFrame({'iter':[i for i in trials.keys()],
                                        'loss': [x['loss'] for x in trials.values()], 
                                        'status': [x['status'] for x in trials.values()], 
                                        'msg': [x['msg'] for x in trials.values()],
                                        'runtime':[x['runtime'] for x in trials.values()]})
            else:
                my_df = pd.DataFrame({'iter':trials.tids,
                                        'loss': [x['loss'] for x in trials.results], 
                                        'status': [x['status'] for x in trials.results], 
                                        'msg': [x['msg'] for x in trials.results],
                                        'runtime':[x['runtime'] for x in trials.results]})
            my_df.to_csv(self.DATA_FOLDER+'/'+log_file_name+'.csv', index = True, header=True)
        except Exception as e:
            print('LOGGING:: ',e)
            print('ERROR at:',self.dataset_id,'Rand:',str(self.seed),'---NO RECORD---')
    def run(self):
        self.i=0
        isFair=True        
        g_csvbest,g_testbest,g_xopt,g_runtime=0,0,'',0
        errorcount,successcount,invalidcount,timeoutcount =0,0,0,0
        n_init_sample=self._init_sample if self.isDaC else (self._init_sample*self.number_candidates)
        #int(self._max_eval*self._init_ratio) #if (BO_type !='Original') else 20
        _max_queue_len=self._max_threads #if (suggest =='rand' and BO_type =='Original') else 1
        #print(n_init_sample,self.ifAllSolution, type(self.ifAllSolution))
        _type=self.BO_optimizer+('-DAC' if self.isDaC else '')
        trials = Trials()
        log_file_name=self.compare_strategy+'-'+str(n_init_sample)+'-'+str(self.number_candidates)+'--'+self.suggest+'_'+_type+'_'+str(self.task_id)+'_'+str(self.dataset_id)+'_'+str(self.seed)+'_'+str(self._sample_sp)+'_'+str(self._slurmlog)
        opt = DACOpt(self.support.search_space, self.objective_func,forbidden=self.support.myforb,
                     conditional=self.support.con
                     ,isDaC=self.isDaC, HPOopitmizer=self.BO_optimizer, random_seed=self.seed, 
                     max_threads=self._max_threads
                     ,eta=self.eta,compare_strategy=self.compare_strategy,
                     max_eval=self._max_eval,hpo_algo=self.suggest,
                     show_message=self.show_message,number_candidates=self.number_candidates
                , timeout= self.time_out,n_init_sp=self._sample_sp
                , n_init_sample=n_init_sample,hpo_trials=trials)
        starttime=time.time()
        opt.start_time=starttime
        opt.timeout=self.time_out
        xopt, fopt, _, eval_count = opt.run()                
        print('XXXX',fopt,RunBenchmark.successcount,'XXXX')
        trials=opt.results if self.BO_type !='hpo' else trials
        #self.anhlog=trials
        self.logging(log_file_name,trials)
        xopt=self.support._getparamflat(xopt)
        xopt= self.support.convertparams(xopt)
        runtime = time.time() - starttime
        _numtest= 3 if self.BO_optimizer=='bo4ml' else 1
        _testLst=[]
	g_testbest=self.test_func(xopt)
        errorcount,successcount,invalidcount,timeoutcount =RunBenchmark.errorcount,RunBenchmark.successcount,RunBenchmark.invalidcount,RunBenchmark.timeoutcount
        test_mean=g_testbest
        self.xopt=xopt
        self.fopt=fopt
        csv_mean= self.fopt
        #print(test_mean,csv_mean,_, eval_count)
        finallog= self.HOME_FOLDER+"/Automl_DACOpt.csv"
        if (os.path.exists(finallog)==False):
            with open(finallog, "a", newline="", encoding="utf-8") as f:    
                wr = csv.writer(f, dialect='excel')
                wr.writerow(['task_id','data_id','multiclass','method','Type','optimizer','random_state','csvmean',
                             'testmean','runtime', 'params','n_init_sample','number_candidates','compare_strategy','max_threads','slurmlog','errorcount','successcount','invalidcount','timeoutcount','isFair'])
        with open(finallog, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow([self.task_id,self.dataset_id,self.multiclass,self.suggest,_type,self.BO_optimizer,
                         self.seed,csv_mean,test_mean,runtime, 
                         self.xopt, n_init_sample, self.number_candidates, self.compare_strategy,self._max_threads,self._slurmlog, 
                         errorcount,successcount,invalidcount,timeoutcount,_testLst])


# In[4]:


#bm.objective_func({'p1': 'none', 'p2': 'decision_tree', 'p14': 'entropy', 'p15': 0.4773909239390577, 'p16': 1.0, 'p17': 'None', 'p18': 0.0, 'p19': 11, 'p20': 5, 'p21': 0.0, 'p3': 'encoding', 'p4': 'no_coalescense', 'p5': 'most_frequent', 'p6': 'power_transformer', 'p7': 'feature_agglomeration', 'p116': 'manhattan', 'p117': 'complete', 'p118': 103, 'p119': 'mean'}

#)


# In[ ]:


if __name__ == '__main__':
    random_state = int(sys.argv[1])
    task_id = int(sys.argv[2])
    isDaC = bool(sys.argv[3])
    BO_optimizer=sys.argv[4]
    number_candidates=int(sys.argv[5])
    _init_sample=int(sys.argv[6])
    compare_strategy=sys.argv[7]
    _slurmlog=sys.argv[8]
    compare_strategy_lst=['stat','highest','rank','mean','medium']
    compare_strategy=compare_strategy_lst[int(compare_strategy)]# if isinstance(compare_strategy,int) else compare_strategy
    '''random_state = 1
    task_id = 3543
    isDaC = True
    BO_optimizer='bo4ml'
    number_candidates=10
    _init_sample=5
    compare_strategy='highest'
    _slurmlog='weraara' '''
    _max_threads=1 
    suggest = 'tpe'
    hpo_prefix='name'
    _max_eval=1000000
    eta=3
    time_out=3600
    show_message=False
    _sample_sp= _init_sample*2
    _startedtime=time.time()
    print('===Ds: ',task_id ,' =Seed: ',random_state,' =Type: ', BO_optimizer, ' = _init_sample:',_init_sample,'===',_slurmlog)
    bm=RunBenchmark(task_id=task_id, random_state=random_state,isDaC= isDaC, eta=eta,
                 compare_strategy=compare_strategy, suggest=suggest,BO_optimizer=BO_optimizer,
                 _slurmlog=_slurmlog,_max_threads=_max_threads,
                    _max_eval=_max_eval,number_candidates=number_candidates,
                    _init_sample=_init_sample,time_out=time_out,
                 _sample_sp=_sample_sp,show_message=show_message)
    #bm=RunBenchmark(task_id,random_state,isDaC,suggest,BO_optimizer,_slurmlog,_max_threads,
    #                _max_eval,ifAllSolution,_init_sample,time_out)
    bm.run()
    print(time.time()-_startedtime)

