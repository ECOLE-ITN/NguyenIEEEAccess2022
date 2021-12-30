from __future__ import absolute_import
from BanditOpt.BO4ML import BO4ML
from BanditOpt.ConditionalSpace import ConditionalSpace
from BanditOpt.ConfigSpace import ConfigSpace
from BanditOpt.Forbidden import Forbidden
from Component.mHyperopt import hyperopt
from Component.mHyperopt import rand, tpe, anneal, atpe, Trials
from BanditOpt import CategoricalParam, FloatParam, Forbidden, \
    IntegerParam, ConfigSpace, ConditionalSpace, AlgorithmChoice, HyperParameter
from BanditOpt.HyperoptConverter import SubToHyperopt, OrginalToHyperopt, ForFullSampling
from BanditOpt.HyperParameter import paramrange, p_paramrange, one_paramrange
import numpy as np
from copy import deepcopy
from collections import Counter
from numpy.random import randint
import math, time
from . import hyperopt as HO
from . import rand,tpe, anneal,atpe
from functools import partial
def InitialModel(self, search_space):
    #np.random.seed(self.seed)
    self.rstate = np.random.RandomState(self.seed) if not hasattr(self, 'rstate') else self.rstate
    self.BOLst=dict()
    # self.trials = Trials()
    kwargs = deepcopy(self.HPO)
    if not hasattr(self, 'isParallel'):
        self.isParallel=False
    if not hasattr(self,'sp_id'):
        self.sp_id=0
    kwargs['isParallel'] =self.isParallel
    kwargs['trials'] = self.trials
    #print(self.sp_id,kwargs )
    self._max_init=self.n_init_sample
    '''if self.hpo_suggest == "tpe":
        _hpo_algo = partial(tpe.suggest, n_startup_jobs=self.n_init_sample,
                            n_EI_candidates=self.n_EI_candidates)
        kwargs['algo'] = _hpo_algo'''
    if self.isFullSearch:
        _totalSP = len(self._LstHPOsearchspace)
        sample_sp = self.n_init_sample / _totalSP if self.sample_sp == None else self.sample_sp
        _max_init = max(self.n_init_sample, sample_sp * _totalSP) #if self.init_ratio < 1 else self.max_eval
        self._max_init=_max_init
        # print("====INIT:", _max_init)
        # if _max_init > self.max_eval and self.max_eval > 0:
        #    raise TypeError("Not Enough budget")
        _step_size = np.floor(_max_init / _totalSP) if self.sample_sp == None else sample_sp
        _max_eval, _imax_eval = 0, 0
        _eval_counted = 0
        self.max_threads = _totalSP if self.max_threads == "max" else self.max_threads
        assert isinstance(self.max_threads, int)
        self._lsstep_size = [max(1, math.floor(x * _step_size)) for x in
                        self._spRatio] if self.sample_sp == None else [_step_size for x in self._spRatio]
        if _max_init > sum(self._lsstep_size):
            _remainsamples = _max_init - sum(self._lsstep_size)
            _most_common = dict(sorted(Counter(dict(zip(range(_totalSP), self._spRatio))).most_common(
                round(_totalSP / self.eta if _totalSP > self.eta else _totalSP))))
            _asum = sum(_most_common.values())
            _most_common = {i: x / _asum for i, x in _most_common.items()}
            _remainBG = dict(Counter(self.rstate.choice(list(_most_common.keys()), replace=True,
                                                      p=list(_most_common.values()), size=_remainsamples)))
            for i, v in _remainBG.items():
                self._lsstep_size[i] = self._lsstep_size[i] + v
        kwargs['timeout'] = self.timeout - (time.time() - self.start_time) if self.timeout != None else None
        for iid, _hposp in enumerate(self._LstHPOsearchspace):
            _hposp = self._LstHPOsearchspace[iid]
            _thisStepsize = 1 if self.shuffle == True else self._lsstep_size[iid]
            kwargs['max_queue_len'] = min(_thisStepsize, self.hpo_max_queue_len)
            kwargs['search_space'] = _hposp
            kwargs['max_evals'] = 0
            kwargs['fix_max_evals'] = 0
            kwargs['rstate']=self.rstate
            #kwargs['n_init_sample'] = 0

            #kwargs['timeout'] = self.timeout - (
            #        time.time() - self.start_time) if self.timeout != None else None
            if _imax_eval > _max_init or (kwargs['timeout'] != None and kwargs['timeout'] <= 0):
                break
            #kwargs['algo'] = rand.suggest
            self.BOLst[iid] = HO.HyperOpt(**kwargs)
    kwargs['rstate'] = self.rstate
    kwargs['search_space'] = search_space
    kwargs['fix_max_evals'] = self.max_eval
    kwargs['timeout'] = self.timeout - (time.time() - self.start_time) if self.timeout != None else None
    self.BO = HO.HyperOpt(**kwargs)
    self.isModelCreated=True
    return
def runBOWithLimitBudget(self, added_budget):
    self.start_time = time.time() #Hack for parallel
    if not hasattr(self, 'isModelCreated') or self.xopt==None:
        self.isModelCreated=False
        self.InitialModel(self.searchspace)
        if not hasattr(self, 'ieval_count'):
            self._eval_counted, self._max_eval,self._imax_eval, self.ieval_count, self.eval_count=0,0,0,0,0
            self.fopt, self.xopt = None, None
        else:
            pass
            #if self.isHyperopt==False:
            #    self._lsstep_size=self.lsstep_size
    else:
        pass
    self.isInitMode=True if  self.ieval_count<self._max_init else False
    Total_max_evals=self.ieval_count+added_budget

    #print('Timeout:', self.timeout)
    while (self.ieval_count < Total_max_evals and ((self.timeout - (time.time() - self.start_time)>0) if self.timeout != None else True)  ):
        _thisStepsize = Total_max_evals-self.ieval_count
        if self.isInitMode:
            #print('______INIT________')  # if self.isInitMode else '*****BO*****')
            if self.isFullSearch:
                #while self._imax_eval<Total_max_evals:
                _randomOrder = [x for x, v in enumerate(self._lsstep_size) if v > 0]
                if len(_randomOrder) > 0:
                    iid = self.rstate.choice(_randomOrder, 1)[0]
                    #print('______INIT________',iid,_randomOrder)
                    try:
                        BO = self.BOLst[iid]
                    except Exception as e:
                        print(e)
                    _thisStepsize = 1 if self.shuffle == True else self._lsstep_size[iid]
                    self._lsstep_size[iid] = self._lsstep_size[iid] - _thisStepsize
                    _max_eval, _imax_eval = self.eval_count, self.ieval_count
                    _max_eval += _thisStepsize
                    _imax_eval += _thisStepsize
                    BO.max_queue_len = min(_thisStepsize, self.hpo_max_queue_len)
                    BO.max_evals = _max_eval
                    BO.fix_max_evals = _imax_eval
                    #BO.rstate=self.rstate
                    # BO.n_init_sample = _imax_eval
                    BO.timeout = self.timeout - (time.time() - self.start_time) if self.timeout != None else None

                    if self.ieval_count > Total_max_evals or (
                            BO.timeout != None and BO.timeout <= 0) or self.isInitMode == False:
                        # print("BREAK")
                        break
                    _results= BO.AddBudget_run(_thisStepsize)#BO.run()
                    self.isInitMode = True if self.ieval_count < self._max_init else False
                    ##self.isInitMode =True if len(self._lsstep_size) > 0 else False
                else:
                    # self.BO.max_evals = _max_eval
                    # self.BO.fix_max_evals = Total_max_evals
                    #self.BO.rstate = self.rstate
                    self.BO.timeout = self.timeout - (
                            time.time() - self.start_time) if self.timeout != None else None
                    _results= self.BO.AddBudget_run(added_budget)
                    self.isInitMode = True if self._imax_eval <= self._max_init else False
            else:
                #self.BO.fix_max_evals = Total_max_evals
                #self.BO.rstate = self.rstate
                _thisStepsize = added_budget
                self.BO.timeout = self.timeout - (time.time() - self.start_time) if self.timeout != None else None
                _results= self.BO.AddBudget_run(added_budget)
                self.isInitMode = True if self._imax_eval < self._max_init else False
        else:
            _thisStepsize = Total_max_evals - self.ieval_count
            #print('**** ====BO Mode=== &&&&&')
            if not hasattr(self, 'isBOMode'):
                _lsthpo_algo = {"rand": rand.suggest, "tpe": tpe.suggest, "atpe": atpe.suggest, "anneal": anneal.suggest}
                if self.hpo_suggest == "tpe":
                    _hpo_algo = partial(tpe.suggest, n_startup_jobs=1,
                                        n_EI_candidates=self.n_EI_candidates)
                else:
                    _hpo_algo=_lsthpo_algo[self.hpo_suggest]
                self.BO.algo= _hpo_algo
                self.isBOMode=True
                self.isInitMode=False
            #self.BO.max_evals = Total_max_evals
            #self.BO.fix_max_evals = Total_max_evals
            self.BO.timeout= self.timeout - (time.time() - self.start_time) if self.timeout != None else None
            _results = self.BO.AddBudget_run(_thisStepsize)
        '''if self.isParallel:
            self.fopt, self.xopt, self._max_eval, self._imax_eval,trials, sp_id, self.rstate = BO.AddBudget_run(_thisStepsize)#BO.run()
        else:
            self.fopt, self.xopt, self._max_eval, self._imax_eval = BO.AddBudget_run(_thisStepsize)  # BO.run()'''
        if self.isParallel:
             self.xopt, self.fopt, self.eval_count, self.ieval_count= _results
        else:
             self.xopt, self.fopt, self.eval_count, self.ieval_count=_results
        self.isInitMode = True if self.ieval_count < self._max_init else False
    '''if (self.isHyperopt):
        _trials = sorted([x for x in self.BO.trials], key=lambda x: x["book_time"])
        self.results = self._save_results(_trials)
        self.trials = _trials
        self.fopt = self.BO.trials.best_trial['result']['loss']
        self.xopt={k:v[0] for k,v in self.BO.trials.best_trial['misc']['vals'].items() if len(v)>0}
        #self.xopt_todict = space_eval(self.searchspace, {k: v for k, v in self.xopt.items()})
        #self.fopt=self.BO.HPO.fopt
        # del _trials'''
    #print('BO: runtime ',time.time()-self.start_time)
    return (self.xopt, self.fopt, self.eval_count, self.ieval_count,self.sp_id,self) if self.isParallel else (self.xopt, self.fopt, self.eval_count, self.ieval_count)

def AddBudget_run(self, add_eval, round_id=1):
    if self.isFullSearch:
        #np.random.seed(self.seed)
        # self.trials = Trials()
        kwargs = deepcopy(self.HPO)
        kwargs['isParallel'] = False
        _max_eval = 0