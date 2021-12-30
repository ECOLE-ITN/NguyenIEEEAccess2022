from __future__ import absolute_import
from joblib import Parallel, delayed
import copy,time, heapq
from collections import OrderedDict
from scipy.stats import friedmanchisquare, ttest_ind, wilcoxon
import numpy as np
from . import BO4ML, Forbidden,ObjectiveFunction, stac, \
    ConfigSpace, ConditionalSpace, Extension, rand, tpe, anneal, atpe, Trials
__author__ = "Duc Anh Nguyen"
BIG_VALUE=9999999999999
class DACOpt(object):
    def __init__(self, search_space: ConfigSpace,
                 obj_func,#surrogate=None,
                 conditional: ConditionalSpace = None,
                 forbidden:Forbidden = None,
                 eta=3,
                 isDaC=True,
                 compare_strategy='Highest',# Median, Mean, Ranktest, highest, Stat
                 SamplingOption="full",
                 HPOopitmizer= "BO4ML",
                 minimize=True,
                 max_eval=None,
                 isFair=True,
                 timeout=None,
                 hpo_prefix='value',
                 max_threads="max",
                 random_seed=None,
                 n_EI_candidates="auto",
                 ifAllSolution=False,
                 number_candidates=None,
                 shuffle=True,
                 min_sp=1,
                 n_init_sp=None, #use for DACOpt: number of samples for initial round per candidate
                 n_init_sample=20,#use for BO: number of init samples
                 hpo_trials=None,
                 hpo_algo= 'tpe',
                 show_message=False):
        #DACOpt: parameter setting
        newObjFunc=ObjectiveFunction.ObjectiveFunction(obj_func,conditional,forbidden,hpo_prefix,minimize)
        self.eta=eta
        self.eval_count = 0
        self.stop_dict = {}
        self.max_threads=max_threads if isinstance(max_threads,int) else 4
        self.HPOopitmizer= HPOopitmizer
        self.start_time=time.time()
        self.max_eval = max_eval
        self.isminize=minimize
        self.timeout=timeout
        self.ifAllSolution=ifAllSolution
        self.n_init_sample=n_init_sample
        self.number_candidates=number_candidates
        self.isFair=isFair
        self.isParallel=True if max_threads>1 else False
        self.shuffle=True if timeout !=None and shuffle==None else shuffle
        isDaC =False if conditional == None or isDaC==False else True
        self.isDaC=isDaC
        self.stat=True if compare_strategy.lower()=='stat' else False
        self.selected_strategy=compare_strategy
        self.show_message=show_message
        n_init_sp=int(n_init_sp if n_init_sp != None else n_init_sample)
        self.n_init_sp = n_init_sp#max(1,np.floor(n_init_sample/(number_candidates if isDaC else 1))))
        self.orgSearchspace = search_space.Combine(conditional, forbidden, isDaC, ifAllSolution=ifAllSolution, random_seed= random_seed,min_sp=min_sp,
                                                   n_init_sp=n_init_sp, max_eval=max_eval, number_candidates=number_candidates)
        #Control parameter:
        self._searchspaceRatio=search_space._ratio
        self.seed = random_seed if random_seed != None else np.random.default_rng().integers(2 ** 31 - 1)
        self.rstate=np.random.RandomState(self.seed)
        self._lsCurrentBest = OrderedDict()
        self._lsOrderdBest = OrderedDict()
        self._lsincumbent = OrderedDict()
        self._lsAllResults=dict()
        self.lseval_count=OrderedDict()
        self.errList=[]
        self.opt = OrderedDict()
        self.DAC_kwargs = OrderedDict()
        self.RoundFeeded=[]
        ###mHyperopt:
        if HPOopitmizer.lower() in ['hyperopt','hpo','bo4ml','bo4automl']:
            hpo_pass_expr_memo_ctrl, hpo_verbose,hpo_max_queue_len,hpo_show_progressbar,hpo_return_argmin,init_ratio  =  None,0,1,True,True,None
            self.isHyperopt=True
            self.BO4ML = dict()
            self.BO4ML['search_space']=search_space
            self.BO4ML['obj_func']=newObjFunc.call
            self.BO4ML['conditional'] =conditional
            self.BO4ML['forbidden'] = forbidden
            self.BO4ML['SearchType']= 'full' if HPOopitmizer.lower() in ('bo4ml','bo4automl') else 'random'
            self.BO4ML['HPOopitmizer'] = 'hpo'
            self.BO4ML['minimize'] = minimize
            self.BO4ML['max_eval'] = np.floor(max_eval/number_candidates) if isDaC else max_eval
            self.BO4ML['timeout'] = timeout
            self.BO4ML['n_init_sample'] = n_init_sample
            self.BO4ML['isFair'] = isFair
            self.BO4ML['random_seed'] = random_seed if random_seed != None else None
            self.BO4ML['hpo_prefix'] = hpo_prefix
            self.BO4ML['hpo_algo'] = hpo_algo
            #self.trials = hpo_trials if hpo_trials != None else Trials()
            self.BO4ML['hpo_trials'] = hpo_trials
            self.BO4ML['hpo_pass_expr_memo_ctrl'] = hpo_pass_expr_memo_ctrl
            self.BO4ML['hpo_verbose'] = hpo_verbose
            self.hpo_max_queue_len = hpo_max_queue_len
            self.BO4ML['hpo_max_queue_len'] = hpo_max_queue_len
            self.BO4ML['hpo_show_progressbar'] = hpo_show_progressbar
            self.BO4ML['hpo_return_argmin']=hpo_return_argmin
            self.BO4ML['ifAllSolution'] = int(n_init_sample) if isinstance(ifAllSolution,bool) else ifAllSolution
            self.BO4ML['sample_sp'] = max(1,n_init_sp/n_init_sp)#number_candidates
            self.BO4ML['max_threads'] = 1
            self.BO4ML['n_EI_candidates'] = n_EI_candidates
            self.BO4ML['shuffle']=shuffle
            if isDaC:
                for i,x in enumerate(self.orgSearchspace):
                    self.DAC_kwargs[i]=copy.deepcopy(self.BO4ML)
                    self.DAC_kwargs[i]['search_space'] = x
                    self.DAC_kwargs[i]['random_seed'] =self.rstate.randint(2 ** 31 - 1)
                    #self.opt[i] = BO4ML(**_DAC_kwargs)
            else:
                self.opt = BO4ML(**self.BO4ML)
        else:
            pass
    def run(self):
        if self.isDaC:
            if self.stat==True:
                return self.ContestProcedurewithStat()
            else:
                return self.ContestProcedure()
        else:
            return self.opt.run()
        print('Finish')
    def ContestProcedurewithStat(self):
        trials=dict()
        _processID = list(self.DAC_kwargs.keys())
        _global_timeout = self.timeout
        iround,num_candidate,num_races=0,self.number_candidates,2
        #_strRound = ('INIT' if iround == 0 else iround)
        _startime = time.time()
        _remainEvals = self.max_eval if self.timeout==None else BIG_VALUE
        _remain_time_this_round = self.timeout if self.timeout != None else None
        while (_remainEvals> 0) and (_remain_time_this_round > 0 if self.timeout != None else True):
            _processID, _lsstep_size, num_candidate = self._createRoundParameters(iround, num_races, num_candidate,_processID)
            if self.show_message:print("Round: ", ('INIT' if iround == 0 else iround), " === ", num_candidate, " search space(s) === runs:", _lsstep_size)
            self.rstate.shuffle(_processID)
            _run_needed = max(1, len(_processID) / self.max_threads)
            _set_localtimeout = (_remain_time_this_round/_run_needed) if self.timeout!=None else None
            self.RunwithBudgetParallel(_processID, _lsstep_size, iround,
                                       _set_localtimeout) if self.isParallel else [
                self.RunwithBudget(x, _lsstep_size, iround, _set_localtimeout) for i, x in
                enumerate(_processID)]
            iround+=1
            _remainEvals = self.max_eval - sum(self.lseval_count.values()) if self.timeout == None else BIG_VALUE
            _remain_time_this_round = self.timeout - (
                    time.time() - _startime) if self.timeout != None else 0
        best_cdid = self.TopHighest(self._lsCurrentBest, 1, Strategy='Highest')
        best_incumbent, best_value = self._lsincumbent[best_cdid[0][0]], self._lsCurrentBest[best_cdid[0][0]]
        _trials = sorted([j for i in [x.trials for x in self.opt.values()] for j in i],
                         key=lambda x: x["book_time"])
        self.results = self._save_results(_trials)
        del _trials
        self.trials = {i: {'search_space': x.searchspace, 'trials': x.trials} for i, x in self.opt.items()}
        # print('Runtime: ',time.time()-self.start_time)
        return best_incumbent, best_value, trials, self.eval_count
    def _createRoundParameters(self, iround, num_races,num_candidate,_processedID):
        if iround == 0:
            self.eval_count = 0
            _processID = list(self.DAC_kwargs.keys())
            #_lsstep_size= [int(max(1, np.floor(x * self.n_init_sp))) for x in self._searchspaceRatio] if self.isFair else [self.n_init_sp] * self.number_candidates
            _lsstep_size = [self.n_init_sp] * self.number_candidates
            for i, x in enumerate(_lsstep_size):
                self.DAC_kwargs[i]['ifAllSolution'] = x if self.isFair and isinstance(self.ifAllSolution,
                                                                                      bool) else self.ifAllSolution
                n_init_sample = self.DAC_kwargs[i]['n_init_sample']
                self.DAC_kwargs[i][
                    'n_init_sample'] = x if self.isFair and self.n_init_sp == n_init_sample else n_init_sample
                # init round has to run seperately to ensure that all candidates samplep
        else:
            if self.stat == False:
                self.eval_count = np.sum([x for x in self.lseval_count.values()])
                if self.timeout == None:
                    max_eval = self.max_eval - self.eval_count
                    eval_race = max_eval / (num_races - iround)
                else:
                    eval_race = self.eval_count / iround + 1
                lsThisRound = self.TopHighest(self._lsCurrentBest, num_candidate,self.selected_strategy,_processedID)
                _processID = sorted([x for x, _ in lsThisRound])
                num_candidate = len(lsThisRound)
                cd_add_eval = int(np.floor(eval_race / num_candidate)) if num_candidate!=0 else 0
                # print('_processID: ',_processID)
                if (num_candidate == 1 and self.timeout == None):
                    remain_eval = self.max_eval - self.eval_count
                    cd_add_eval = max(cd_add_eval, remain_eval)
                _lsstep_size = [(cd_add_eval if x in _processID else 0) for x in range(0, self.number_candidates)]
            else:
                '''Perform statistical test here:'''
                _ = self._updateAllresults(_processedID)
                number_of_elements = min(
                    [len(x) for i, x in self._lsAllResults.items() if i in _processedID]) - self.n_init_sample
                _min_element=1
                _default_stepSize = _min_element- number_of_elements if number_of_elements<_min_element else 1
                if num_candidate==1:
                    _default_stepSize=self.max_eval-sum(self.lseval_count.values()) if self.timeout==None else BIG_VALUE
                _processID = self.StatTest(_processedID,0.05,min_element=_min_element) if len(_processedID)>1 else _processedID
                num_candidate =len(_processID)
                #_lsstep_size = [_default_stepSize for x in range(0, self.number_candidates) if x in _processID]
                _lsstep_size = [(_default_stepSize if x in _processID else 0) for x in range(0, self.number_candidates)]
        return  _processID, _lsstep_size, num_candidate
    def ContestProcedure(self):
        trials = dict()
        lsRace = self.calculateSH()
        num_races = len(lsRace)
        _global_timeout=self.timeout
        #_avg_RuntimePerRound=self.timeout/num_races
        #print(_avg_RuntimePerRound)
        _processID=None
        for iround, num_candidate in lsRace.items():
            _strRound=('INIT' if iround == 0 else iround)
            _startime=time.time()
            _processedID=_processID if _processID!=None else None
            _processID, _lsstep_size, num_candidate= self._createRoundParameters(iround, num_races,num_candidate,_processedID)
            if self.show_message:print("Start Round: ", _strRound, " === ", num_candidate, " search space(s) === runs:",_lsstep_size)
            _lsstep_size_temp = _lsstep_size
            _before= [x for i,x in self.lseval_count.items() if i in _processID]if len(self.lseval_count)>0 else [0]*num_candidate
            _avg_RuntimePerRound,_remain_time_this_round=None, None
            if self.timeout != None:
                self.timeout = _global_timeout- (time.time() - self.start_time)
                _avg_RuntimePerRound = (self.timeout / (num_races - iround)) #if iround!=0 else self.timeout
                _remain_time_this_round = _avg_RuntimePerRound if iround!=0 else self.timeout
                if self.timeout <= 0:
                    print('DACOpt message: TimeOut')
                    break
            _run_needed=max(1,len(_processID)/self.max_threads)
            _init_counted = [(self.lseval_count[x] if x in _processID else 0) for x in range(0, self.number_candidates)] if len(self.lseval_count) > 0 else [0] * num_candidate
            _repeated=0
            while ((sum(_lsstep_size_temp) if isinstance(_lsstep_size_temp,list) else _lsstep_size_temp)>0) and (_remain_time_this_round > 0 if self.timeout !=None else True):
                self.rstate.shuffle(_processID)
                _set_localtimeout=(_remain_time_this_round/_run_needed) if self.timeout!=None else None
                #print('_set_localtimeout',_set_localtimeout)
                self.RunwithBudgetParallel(_processID, _lsstep_size_temp, iround,_set_localtimeout) if self.isParallel else [
                    self.RunwithBudget(x, _lsstep_size_temp, iround,_set_localtimeout) for i, x in enumerate(_processID)]
                #time.sleep(3)
                _repeated+=1
                if self.timeout!=None:
                    _timeUsed = time.time() - _startime
                    _remain_time_this_round=_avg_RuntimePerRound-_timeUsed
                    _timeUsedRatio=(_remain_time_this_round/_timeUsed) if _timeUsed>0 else 0
                    #update real number of runs:
                    _after_counted=[(self.lseval_count[x] if x in _processID else 0) for x in range(0, self.number_candidates)] if len(self.lseval_count) > 0 else [0] * num_candidate
                    #_after_counted=[x for i, x in self.lseval_count.items()] if len(self.lseval_count) > 0 else [0] * num_candidate
                    _lsstep_size_temp=[x - y for x, y in zip(_after_counted, _init_counted)]
                    if self.show_message:print('Round',_strRound, 'message: REAL evals counted:',[(x,y) for x,y in zip(range(0, self.number_candidates), _lsstep_size_temp) if x in _processID])
                    if _remain_time_this_round>=0:
                        _avg_runtime_1eval=(_timeUsed/np.mean(_lsstep_size_temp))
                        _time_per_thread = _remain_time_this_round / (_run_needed+0.2) #-20% for other stuffs
                        #print('CHECK ',_remain_time_this_round, _avg_runtime_1eval, np.mean(_lsstep_size_temp))
                        _cd_added=int(np.floor(_remain_time_this_round/_avg_runtime_1eval))
                        _lsstep_size_temp = [(_cd_added if x in _processID else 0) for x in range(0, self.number_candidates)]
                        #print(_lsstep_size_temp)
                        #_lsstep_size_temp = [int(np.floor(x * _timeUsedRatio)) for x in _lsstep_size_temp] if isinstance(_lsstep_size_temp,list) else int(np.floor(_lsstep_size_temp*_timeUsedRatio))
                        #print('Computation is cheaper than estimated, an additional budgets is added to this round...')
                        if self.show_message:print('Round', _strRound, 'message: still some time left, an additional budgets is added to this round:', _lsstep_size_temp)
                    else:
                        _lsstep_size_temp = 0
                        if self.show_message:
                            print('Round', _strRound, 'message: Timeout for this round-- Runtime counted: ', time.time() - self.start_time)
                            print(('Go to the next round') if iround<num_races-1 else ('Finished'))
                        continue
                    _remain_time_this_round = _avg_RuntimePerRound - (
                            time.time() - _startime) if self.timeout != None else 0
                else:
                    _lsstep_size_temp = 0
            _=self._updateAllresults(_processID)
        try:
            best_cdid = self.TopHighest(self._lsCurrentBest,1,Strategy='Highest')
            best_incumbent, best_value = self._lsincumbent[best_cdid[0][0]], self._lsCurrentBest[best_cdid[0][0]]
            _trials = sorted([j for i in [x.trials for x in self.opt.values()] for j in i],
                             key=lambda x: x["book_time"])
            self.results = self._save_results(_trials)
            del _trials
            self.trials = {i: {'search_space': x.searchspace, 'trials': x.trials} for i, x in self.opt.items()}
        except:
            best_incumbent, best_value=self._Conclusion()
        #print('Runtime: ',time.time()-self.start_time)
        return best_incumbent, best_value, self.trials, self.eval_count
    def _Conclusion(self):
        for sp_id,obj in self.opt.items():
            xcatch = [x['loss'] for x in obj.BO.trials.results if x['status'] == 'ok']
            ieval_count = len(xcatch)
            eval_count = len([x['loss'] for x in obj.BO.trials.results])
            fopt = min(xcatch) if ieval_count > 0 else None
            if not hasattr(obj, 'fmin'):
                obj.fmin = None
            xopt = obj.fmin
            obj.fopt = fopt
            obj.eval_count = eval_count
            obj.ieval_count = ieval_count
            obj.eval_hist = xcatch
            self._lsincumbent[sp_id] = xopt
            self.lseval_count[sp_id] = ieval_count
            if fopt != None:
                self._lsCurrentBest[sp_id] = fopt
        best_cdid = self.TopHighest(self._lsCurrentBest, 1,Strategy='Highest')
        best_incumbent, best_value = self._lsincumbent[best_cdid[0][0]], self._lsCurrentBest[best_cdid[0][0]]
        _trials = sorted([j for i in [x.trials for x in self.opt.values()] for j in i],
                         key=lambda x: x["book_time"])
        self.results = self._save_results(_trials)
        del _trials
        self.trials = {i: {'search_space': x.searchspace, 'trials': x.trials} for i, x in self.opt.items()}
        return best_incumbent, best_value
    def _updateAllresults(self, processIDs):
        processIDs=self.opt.keys() if processIDs==None  else processIDs
        for ids in processIDs:
            _trials_results = self.opt[ids].trials.results
            self._lsAllResults[ids] = [x['loss'] for x in _trials_results if x['status'] == 'ok']
        return
    def RunwithBudget(self,sp_id,budgets, round_id, timeout=None):
        #print(sp_id,budget, round_id)
        budget=budgets[sp_id] if isinstance(budgets,list) else budgets
        xopt,fopt=None,None
        try:
            _imax_eval=0
            if round_id == 0 and (str(round_id)+'-'+str(sp_id)) not in self.RoundFeeded:
                self.DAC_kwargs[sp_id]['max_eval'] = int(budget)
                self.DAC_kwargs[sp_id]['timeout'] = timeout
                self.opt[sp_id] = BO4ML(**self.DAC_kwargs[sp_id])
                #xopt, fopt,_max_eval, _imax_eval = self.opt[sp_id].runBOWithLimitBudget(int(budget))
            else:
                self.DAC_kwargs[sp_id]['timeout'] = timeout
                self.DAC_kwargs[sp_id]['max_eval'] += int(budget)
                self.opt[sp_id].max_eval+= int(budget)
                self.opt[sp_id].timeout=timeout
            self.RoundFeeded.append(str(round_id)+'-'+str(sp_id))
            xopt, fopt,_max_eval, _imax_eval = self.opt[sp_id].runBOWithLimitBudget(int(budget))
            self.lseval_count[sp_id]=_imax_eval
            #_trials_results = self.opt[sp_id].trials.results
            #self._lsAllResults[sp_id] = [x['loss'] for x in _trials_results if x['status'] == 'ok']
            #self.opt[sp_id]['ieval_count'] = ieval_count
            print('New message::: Round-',round_id,' --candidate ID-', str(sp_id),' -- add:',budget, ' --best-found value: ', str(fopt))
        except Exception as e:
            self.errList.append(sp_id)
            try:
                xcatch = [x['loss'] for x in self.opt[sp_id].BO.trials.results if x['status'] == 'ok']
                ieval_count = len(xcatch)
                eval_count = len([x['loss'] for x in self.opt[sp_id].BO.trials.results])
                fopt = min(xcatch) if ieval_count > 0 else None
                if not hasattr(self.opt[sp_id], 'fmin'):
                    self.opt[sp_id].fmin = None
                xopt = self.opt[sp_id].fmin
                self.opt[sp_id].fopt = fopt
                self.opt[sp_id].eval_count = eval_count
                self.opt[sp_id].ieval_count = ieval_count
                self.opt[sp_id].eval_hist = xcatch
            except Exception as e2:
                print('error:',e2)
            print('New ERROR:::Round-',round_id,' -- Candidate ID-', str(sp_id), '--msg:', e)
        self._lsincumbent[sp_id] = xopt
        if fopt != None:
            self._lsCurrentBest[sp_id] = fopt
        #
    def RunwithBudgetParallel(self,sp_ids,budgetLst, round_id,timeout=None):
        try:
            _imax_eval=0
            if round_id == 0 and round_id not in self.RoundFeeded:
                for sp_id in sp_ids:
                    budget=int(budgetLst[sp_id]) if isinstance(budgetLst,list) else int(budgetLst)
                    self.DAC_kwargs[sp_id]['max_eval'] = budget
                    self.DAC_kwargs[sp_id]['timeout'] = timeout
                    self.opt[sp_id] = BO4ML(**self.DAC_kwargs[sp_id])
                    self.opt[sp_id].start_time=time.time()
                    self.opt[sp_id].isParallel=True
                    self.opt[sp_id].sp_id=sp_id
                    self.opt[sp_id].lsstep_size=None
                #xopt, fopt,_max_eval, _imax_eval = self.opt[sp_id].runBOWithLimitBudget(int(budget))
            else:
                for sp_id in sp_ids:
                    budget = int(budgetLst[sp_id]) if isinstance(budgetLst, list) else int(budgetLst)
                    self.DAC_kwargs[sp_id]['max_eval'] += budget
                    self.DAC_kwargs[sp_id]['timeout'] = timeout
                    self.opt[sp_id].max_eval+= budget
                    self.opt[sp_id].timeout = timeout
                    #self.opt[sp_id].isParallel = True
                    #self.opt[sp_id].sp_id=sp_id
                #xopt, fopt,_max_eval, _imax_eval = self.opt[sp_id].runBOWithLimitBudget(int(budget))
            self.RoundFeeded.append(round_id)
            #TODO: needs to kill process aferward
            #https://stackoverflow.com/questions/67495271/joblib-parallel-doesnt-terminate-processes
            #https://github.com/joblib/joblib/issues/945
            #current_process = psutil.Process()
            #subproc_before = set([p.pid for p in current_process.children(recursive=True)])
            _return=None
            _start = time.time()
            _return = Parallel(n_jobs=self.max_threads)(
                delayed(x.runBOWithLimitBudget)(int(budgetLst[i] if isinstance(budgetLst, list) else budgetLst)) for
                i, x in self.opt.items() if i in sp_ids)
            _ = self._updatebyParallel(sp_ids, _return) if _return != None else None
            print('New message::: Round-', round_id, ' --candidate ID- budget',
                  str([str(x) + '-' + str(budgetLst[x] if
                                          isinstance(
                                              budgetLst,
                                              list) else budgetLst)
                       for x in sp_ids]),
                  ' --best-found value: ',
                  str([str(i) + '-' + str(v) for i, v in self._lsCurrentBest.items() if i in sp_ids]))
            del _return
        #print(time.time()-_start)
        except Exception as e:
            print('New ERROR:::Round-',round_id, '--msg:', e)

    def calculateSH(self) -> OrderedDict():
        lsEval = OrderedDict()
        if self.stat==False:
            remain_candidate = self.number_candidates
            ratio = 1 / self.eta
            a = 0
            lsEval[a] = remain_candidate
            while remain_candidate > 1:
                a += 1
                remain_candidate = np.ceil(remain_candidate * ratio)
                lsEval[a] = int(remain_candidate)
        else:
            remain_candidate = self.number_candidates
            lsEval[0] = remain_candidate
            lsEval[1] = remain_candidate
        return lsEval
    def TopHighest(self,_lsCurrentBest,num_candidate, Strategy='Highest',processedID=None):
        _reverse=False if self.isminize else True
        Strategy=Strategy.lower()
        if Strategy in ('median','mean','rank'):
            _processedID=processedID if processedID!= None else self.opt.keys()
            _ = self._updateAllresults(_processedID)
            if Strategy =="rank":
                number_of_elements = min(
                    [len(x) for i, x in self._lsAllResults.items() if i in processedID])
                _samples=[]
                for idx in processedID:
                    _samples.append([x for x in self._lsAllResults[idx][-number_of_elements:]])
                statistic, p_value, ranking, rank_cmp = stac.friedman_test(*_samples)
                _lsCurrentBest = {key: ranking[i] for i, key in enumerate(processedID)}
            else:
                _npfunc=np.median if Strategy=="median" else np.mean
                _lsCurrentBest= {i:_npfunc(x) for i,x in self._lsAllResults.items() if i in _processedID}
            if self.show_message:print('Computed ',Strategy,' values:',_lsCurrentBest)
        else:
            _lsCurrentBest=self._lsCurrentBest
        lsThisRound = list(OrderedDict(sorted(_lsCurrentBest.items(), key=lambda item: item[1],
                                              reverse=_reverse)).items())[:num_candidate]
        _worstValue= round((max if self.isminize else min)([x for i,x in lsThisRound]),5)
        if num_candidate > 1:
            #If there are candidates with extract the same values
            lsThisRound.extend([(i,x) for i,x in _lsCurrentBest.items() if (round(x,5)<=  _worstValue if self.isminize else round(x,5)>=  _worstValue) and i not in [_[0] for _ in lsThisRound]])
        else:
            pass
        '''if len(lsThisRound)>1 and num_candidate==1:
            _newList=lsThisRound
            for _strategy in [x for x in ('Highest', 'Ranktest', 'Mean', 'Median') if x !=Strategy]:
                _newList=self.TopHighest(None, 1,Strategy=_strategy,processedID=lsThisRound)'''
        return lsThisRound
    def StatTest(self, ids, p=0.05,number_of_elements=30, min_element=3):
        _results=[]
        _ids_sorted=sorted(ids)
        _ = {_name: _id for _id, _name in enumerate(ids)}
        _revered=True if self.isminize else False
        _minOrMax=np.min if self.isminize else np.max
        number_of_elements=min([len(x) for i,x in self._lsAllResults.items() if i in ids])-self.n_init_sample
        if number_of_elements< min_element:
            print('StatTest: NOT enough samples to be tested')
            return ids
        #print(ids,number_of_elements)
        #print([(i,len(x)) for i,x in self._lsAllResults.items() if i in ids])
        for _index in _ids_sorted:
            #_temp = [x for x in self._lsAllResults[_index]]#sorted(self._lsAllResults[_index], reverse=_revered)[:number_of_elements]
            #_temp = heapq.nsmallest(number_of_elements, self._lsAllResults[_index]) if self.isminize else heapq.nlargest(number_of_elements, self._lsAllResults[_index])
            _avg_init_step=_minOrMax(self._lsAllResults[_index][:-number_of_elements])
            _temp=[_avg_init_step]
            _temp.extend(self._lsAllResults[_index][-number_of_elements:])
            #print(_index,_temp)
            _results.append(_temp)
        _statappr = 'FriedmanTest'
        if len(ids)>2:
            _, _p = friedmanchisquare(*_results)
        else:
            _statappr = 'wilcoxon'
            _,_p= wilcoxon(*_results)
        ids_in_Str = [str(i) for i in _ids_sorted]
        _goodnessList = _ids_sorted
        if _p<=p:
            if self.show_message: print('Significant different by{} - stac:{} - p:{}'.format(_statappr,_, _p))
            statistic, p_value, ranking, rank_cmp = stac.friedman_test(*_results)
            ranks = {key: ranking[i] for i, key in enumerate(ids_in_Str)}
            _best_candidate=ids_in_Str[(np.argmin(ranking) if self.isminize else np.argmax(ranking))]
            #print(_best_candidate,ranks)
            comparisons, z_values, p_values, adj_p_values = stac.holm_test(ranks, control=_best_candidate)
            adj_p_values = np.asarray(adj_p_values)
            rank_ordered = [k for k, v in sorted(ranks.items(), reverse=_revered, key=lambda item: item[1])]
            rank_ordered.remove(_best_candidate)
            _goodnessList= [_best_candidate]+[y for x, y in zip(adj_p_values, rank_ordered) if x >= p]
            if self.show_message:
                if len(_goodnessList)==len(ids):
                    print('Keep All: NO significant different to the best (Can.NO-{}) according to a post-hoc Holm test'.format(_best_candidate))
                    print(
                        [(x, (True if x < p else False), y) for x, y in zip(adj_p_values, comparisons)])
                else:
                    print(
                        [(x, (True if x < p else False), y) for x, y in zip(adj_p_values, comparisons)])
            '''print(adj_p_values)
            print(rank_ordered)
            print(comparisons)
            print([(x,(True if x<p else False),y,z) for x,y,z in zip(adj_p_values,comparisons, rank_ordered)])'''
        else:
            if self.show_message:print('No different by FriedmanTest', _p,_)
        #print(_goodnessList if self.show_message else None)
        return [int(x) for x in _goodnessList]

        #return self.rstate.choice(ids,replace=False,size=max(len(ids)- (np.random.choice([0,1])),1))

    def _critical_distance(alpha, k, n):
        """
        Determines the critical distance for the Nemenyi test with infinite degrees of freedom.
        """
        from statsmodels.stats.libqsturng import qsturng
        return qsturng(1 - alpha, k, np.inf) * np.sqrt(k * (k + 1) / (12 * n))
    def _updatebyParallel(self, _lstId, _return: tuple):
        #print(_lstId)
        assert len(_lstId) == len(_return)
        for i in _lstId:
            _thissp = [x for x in _return if x[4] == i][0]
            self.opt[i]=_thissp[5]
            self.lseval_count[i] = _thissp[3]
            #_trials_results=_thissp[5].trials.results
            #self._lsAllResults[i]=[x['loss'] for x in _trials_results if x['status'] == 'ok']
            #del _trials_results
            '''self.opt[i]._imax_eval =_thissp[3]
            self.opt[i].ieval_count = _thissp[3]
            self.opt[i].max_evals=_thissp[2]
            self.opt[i]._max_eval=_thissp[2]
            self.opt[i].eval_count = _thissp[2]
            self.opt[i].trials = _thissp[4]
            # x.isParallel = False
            
            self.opt[i].lsstep_size=_thissp[7]
            self.opt[i].rstate=_thissp[6]'''
            if np.isnan(_thissp[1]) == False:
                _thisMin = float(_thissp[1])
                self._lsCurrentBest[i] = _thisMin
                self._lsincumbent[i] = _thissp[0]
            else:
                print("Please check id ", i, " got NaN value", _thissp[0], _thissp[1])
        self.eval_count = np.sum([x.ieval_count for _, x in self.opt.items()])

        # self.trials=dict(enumerate([x[4] for x in _return]))
        # self._lsCurrentBest=dict(enumerate([x[1] for x in _return]))
        # self._lsincumbent=dict(enumerate([x[0] for x in _return]))
        return None
    @staticmethod
    def _save_results(trials: list):
        return dict(enumerate([x['result'] for x in trials]))

    '''@staticmethod
    def create_trials(orgtrials, trials: list):
        newtrials = orgtrials
        tid = max([trial['tid'] for trial in newtrials.trials]) if len(newtrials) > 0 else -1
        for trial in trials:
            tid = tid + 1 if tid >= 0 else 0
            hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
            hyperopt_trial[0] = trial
            hyperopt_trial[0]['tid'] = tid
            hyperopt_trial[0]['misc']['tid'] = tid
            for key in hyperopt_trial[0]['misc']['idxs'].keys():
                oldVal = hyperopt_trial[0]['misc']['idxs'][key]
                if len(oldVal) > 0:
                    hyperopt_trial[0]['misc']['idxs'][key] = [tid]
            newtrials.insert_trial_docs(hyperopt_trial)
            newtrials.refresh()
        return newtrials

    @staticmethod
    def merge_trials(trials1, trials2):
        newtrials = trials1
        max_tid = max([trial['tid'] for trial in newtrials.trials]) if len(newtrials) > 0 else -1
        tid = max_tid
        for trial in trials2:
            if 1 == 1:
                # if(trial['misc']['vals'] not in [x['misc']['vals'] for x in newtrials]):
                # tid = trial['tid'] + max_tid + 1
                tid = tid + 1
                hyperopt_trial = Trials().new_trial_docs(
                    tids=[None],
                    specs=[None],
                    results=[None],
                    miscs=[None])
                hyperopt_trial[0] = trial
                hyperopt_trial[0]['tid'] = tid
                hyperopt_trial[0]['misc']['tid'] = tid
                for key in hyperopt_trial[0]['misc']['idxs'].keys():
                    oldVal = hyperopt_trial[0]['misc']['idxs'][key]
                    if len(oldVal) > 0:
                        hyperopt_trial[0]['misc']['idxs'][key] = [tid]
                newtrials.insert_trial_docs(hyperopt_trial)
                newtrials.refresh()
        return newtrials
        '''
if __name__ == '__main__':
    from BanditOpt.ConditionalSpace import ConditionalSpace
    from BanditOpt.ConfigSpace import ConfigSpace
    from BanditOpt.Forbidden import Forbidden
    from BanditOpt import CategoricalParam, FloatParam, Forbidden, \
        IntegerParam, ConfigSpace, ConditionalSpace
    from BanditOpt.HyperoptConverter import SubToHyperopt, OrginalToHyperopt, ForFullSampling
    np.random.seed(1)
    cs = ConfigSpace()
    alg_namestr = CategoricalParam(["SVM", "RF", 'LR', 'DT'], "alg_namestr")
    test = CategoricalParam(("A", "B"), "test", default="A")
    testCD = CategoricalParam(("C", "D"), "testCD", default="C")
    C = FloatParam([1e-2, 100], "C")
    degree = IntegerParam([([1, 2], 0.1), ([3, 5], .44), [6, 10], 12], 'degree')
    f = FloatParam([(0.01, 0.5), [0.02, 100]], "testf")
    con = ConditionalSpace("test")
    # arange=range(1, 50, 2)
    abc = CategoricalParam([x for x in range(1, 50, 2)], "abc")
    cs.add_multiparameter([alg_namestr, test, C, degree, f, abc, testCD])
    con.addConditional(test, alg_namestr, "SVM")
    con.addMutilConditional([test, degree], alg_namestr, "RF")
    fobr = Forbidden()
    # fobr.addForbidden(abc, 5, alg_namestr, "SVM")
    fobr.addForbidden(test, 'A', abc, 5)
    fobr.addForbidden(test, 'B', abc, 7)
    fobr.addForbidden(testCD, 'C', abc, 1)
    lsSpace = cs.Combine(con, fobr, isBandit=True)
    lsSpace

