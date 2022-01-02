import copy

from . import HyperParameter, p_paramrange, ConfigSpace, ConditionalSpace, Forbidden,\
    AlgorithmChoice, CategoricalParam, IntegerParam, FloatParam
import numpy as np
from copy import deepcopy
from collections import OrderedDict, Counter
from typing import Union, List, Dict, Optional
from numpy.random import randint
import itertools, collections, math, time
from . import ParamsExt

def Combine(self, Conditional: ConditionalSpace = None, Forbidden: Forbidden = None,
            isBandit: bool = True,sp_cluster=1, ifAllSolution=False, random_seed=0, min_sp=3,
            n_init_sp=None, max_eval=500, init_sample=10, number_candidates=10,sample_sp=5, init_ratio=0.2) -> List[HyperParameter]:
    isDaC=isBandit
    self._listconditional = Conditional
    self._listForbidden = Forbidden
    if (isDaC == True):
        if ifAllSolution==True:
            return self._conditionalfree(Conditional, Forbidden, 0, ifAllSolution)
        else:
            return self._combinewithconditional(Conditional,Forbidden,
                                                random_seed, min_sp, init_sample, number_candidates)
    else:
        #lsSpace = []
        for k, v in self._hyperparameters.items():
            self._hyperparameters[k]=ParamsExt.updateHyperparameterforBO4ML(v)
        self._ratio = None
        return self


def _combinewithconditional(self, cons: ConditionalSpace = None, forb:Forbidden =None, random_seed=0, min_sp=3,
                           init_sample=10, number_candidates=10) -> List[
    HyperParameter]:
    #np.random.seed(random_seed)
    rstate = np.random.RandomState(random_seed)
    _defratio=0.5
    _min_sp = min_sp if min_sp != None else max(1,int(np.floor(init_sample / number_candidates)))
    #_max_sp = int(np.floor(max_eval * _defratio) / number_candidates) if ifAllSolution == False else ifAllSolution
    _max_sp = number_candidates #if ifAllSolution == False else ifAllSolution
    _max_sp = _max_sp + 1 if _max_sp == _min_sp else _max_sp
    lsParentName, lsChildEffect, lsFinalSP, lsVarNameinCons, childList = [], [], [], [], []
    listParam = {i: k for i, k in self._hyperparameters.items()}
    for i, con in cons.conditional.items():
        if (con[1] not in lsVarNameinCons):
            lsVarNameinCons.append(con[1])
        if con[0] not in childList:
            childList.append(con[0])
    lsVarNameinCons = [x for x in lsVarNameinCons if x not in childList]
    lsOneNode = [x for x in listParam.keys() if x not in (lsVarNameinCons + childList)]
    ##List out the branches which have values with no conditional
    _temp,_test, _test2, _ori_bounds  = [],dict(), dict(), dict()
    for vName in lsVarNameinCons + lsOneNode:
        # If the current node is algorithmChoice, listing by bounds, otherwise list based on conditional
        # if isinstance(self._hyperparameters[vName],AlgorithmChoice):
        _thisNode = []
        _ori_bounds[vName]=[x.bounds for x in self._hyperparameters[vName].bounds]
        if isinstance(self._hyperparameters[vName], AlgorithmChoice):
            if len(self._hyperparameters[vName].bounds) > 1:
                _thisNode += [x.bounds for x in self._hyperparameters[vName].bounds]
                _test2[vName] = _thisNode#[i for j in _thisNode for i in j]
            else:
                _thisNode += [x for x in self._hyperparameters[vName].bounds[0].bounds]
                _test[vName] = _thisNode
        else:
            _thisNode = [x.bounds for x in self._hyperparameters[vName].bounds]
            _test2[vName] = _thisNode
    _allBounds = dict()
    for i, x in _test2.items():
        _allBounds[i] = [j for i in x for j in i]
    for i, x in _test.items():
        _allBounds[i] = x
    _numberofgroups = {i: [*range(1, len(v) + 1)] if i in lsVarNameinCons else [1] for i, v in _test.items()}
    _numberofitems,_lsParentName,_splitStrategy = dict(),[],[]
    #_tmax_sp = n_init_sp if n_init_sp != None else _max_sp
    _tmax_sp=_max_sp
    #Anh: 16/12/2021:
    _min_sp=_tmax_sp
    _numberofitems={i:len(x) for i,x in _test.items()}
    for i, x in _test2.items():
        if isinstance(self._hyperparameters[i], (AlgorithmChoice)):
            _numberofgroups[i] = [*range(1, len(x) + 1)]
        else:
            _numberofgroups[i] = [1]
        _numberofitems[i]=len([j for i in x for j in i])
    _ = [x for x in itertools.product(*list(_numberofgroups.values())) if np.product(x) in [*range(_min_sp, _tmax_sp + 1)]]
    if len(_) < 1:
        _numberofgroups = {i: [*range(1, v + 1)] for i, v in _numberofitems.items()}
    _splitStrategy = [x for x in itertools.product(*list(_numberofgroups.values())) if np.product(x) == _tmax_sp]
    while len(_splitStrategy) < 1:
        _tmax_sp = _tmax_sp - 1
        _splitStrategy = [x for x in itertools.product(*list(_numberofgroups.values())) if
                              np.product(x) == _tmax_sp]
    if (len(_splitStrategy) < 1):
        raise TypeError("No spliting solution")
    _tarr, _ibest, _ibestValue = [], [], 0.00
    _i=[x for x in _numberofitems.values()]
    for i, x in enumerate(_splitStrategy):
        _pArr = 0.00
        for _, _x in enumerate(x):
            _nBounds = _i[_]
            _pro = 1 - (_x / _nBounds)
            _pArr = _pArr + _pro
        if _ibestValue == _pArr:
            _ibest.append(i)
        elif _ibestValue < _pArr:
            _ibestValue = _pArr
            _ibest = [i]
        else:
            pass
        _tarr.append(_pArr)
    _param_ori = []
    _splitStrategy = [_splitStrategy[i] for i in _ibest]
    if len(_splitStrategy) > 1:
        _tarr = [_tarr[i] for i in _ibest]
        _t = np.sum(_tarr)
        _p = [math.floor((x / _t) * 1000) / 1000 for x in _tarr[:-1]]
        _p.append(1 - sum(_p))
        _choosenstr = _splitStrategy[rstate.choice(len(_splitStrategy), p=_p)]
    else:
        _choosenstr = _splitStrategy[0]
    for i, x in enumerate(_choosenstr):
        _key = list(_numberofgroups)[i]
        _thisnode = _ori_bounds[_key]
        _regrouped=[]
        _thisseed=rstate.randint(2 ** 31 - 1)
        _regrouped=ParamsExt.regroup(_thisnode,_togroup=x,seed=_thisseed)
        for _bounds in _regrouped:
            _lsParentName.append([_key, _bounds])
    newlsParentName = []
    for item, count in collections.Counter([x[0] for x in _lsParentName]).items():
        if (count == 1):
            newlsParentName.extend([[x[0], x[1]] for x in _lsParentName if x[0] == item])
        else:
            temp = [[x[0], len(x[1]), x[1]] for x in _lsParentName if x[0] == item]
            temp.sort(reverse=False)
            feeded = []
            for index, rootvalue in enumerate(temp):
                # print(index,rootvalue)
                flag = False
                for value in temp[index + 1:]:
                    abc = set(rootvalue[2]).intersection(set(value[2]))
                    #abc=np.intersect1d(rootvalue[2],value[2])
                    if (len(abc) > 0):
                        flag = True
                    #if len(np.intersect1d(rootvalue[2],[item for sublist in feeded for item in sublist]))>0:
                    if (len(set(rootvalue[2]).intersection([item for sublist in feeded for item in sublist])) > 0):
                        flag = True
                if (flag == True):
                    for i in rootvalue[2]:
                        if (i not in ([item for sublist in feeded for item in sublist])):
                            newlsParentName.append([rootvalue[0], [i]])
                            feeded.append([i])
                else:
                    dif =sorted(list(set(rootvalue[2]).difference([item for sublist in feeded for item in sublist])), key=lambda x: str(x))
                    rstate.shuffle(dif)
                    #dif = list(np.setdiff1d(rootvalue[2],[item for sublist in feeded for item in sublist]))
                    newlsParentName.append([rootvalue[0], dif])
                    feeded.append(dif)
    for item in newlsParentName:
        # con=
        _thisnode = item[1]  # [j for i in [x for x in item[1]] for j in i]
        for con in [x for x in cons.conditional.values() if
                    x[1] == item[0] and len(set(x[2]).intersection(_thisnode))]:
            lsChildEffect.append([str(con[1]) + "_" + "".join(_thisnode), con[0]])
    lsSearchSpace = self.listoutAllBranches(lsVarNameinCons, lsChildEffect, newlsParentName)
    _returnSubSpaces,_returnCons,_returnForb=[],[],[]
    _ratio=[]
    for x in lsSearchSpace:
        _configSpace=ConfigSpace()
        _configSpace.add_multiparameter([i for i in x])
        #fix bug smaller conditional space
        _lsVarNames=np.unique([i.var_name for i in x])
        _newcons=ConditionalSpace('n-name')
        _newcons.conditional=OrderedDict({i:v for i,v in cons.conditional.items() if v[0] in _lsVarNames and v[1] in _lsVarNames})
        _newcons.AllConditional = OrderedDict({i: v for i, v in cons.AllConditional.items() if
                                v[0] in _lsVarNames or v[1] in _lsVarNames})
        _numberofPipeline=np.product([len(i.bounds) for i in x if isinstance(i,AlgorithmChoice)])
        if forb != None:
            _newforb=Forbidden()
            _newforb.forbList=OrderedDict({i:v for i,v in forb.forbList.items() if v.left in _lsVarNames and v.right in _lsVarNames})
            _returnForb.append(_newforb)
        else:
            _returnForb.append(None)
        _numberofParameters=len(x)
        _ratio.append(_numberofPipeline)
        _returnSubSpaces.append(_configSpace)
        _returnCons.append(_newcons)

    _avgratio=np.mean(_ratio)
    _ratio=[(x/_avgratio) for x in _ratio]
    self._ratio=_ratio
    self._returnCons=_returnCons
    self._returnForb=_returnForb
    return _returnSubSpaces
def listoutAllBranches(self, lsvarname, childeffect, lsparentname) -> List[HyperParameter]:
    np.random.RandomState(24)
    # hp = copy.deepcopy(self._hyperparameters)
    hp = self._hyperparameters
    temp_hpi, lsBranches = [], []
    childList = [x[1] for x in childeffect]
    lsvarname = list(lsvarname)
    lsOneNode = [x for x in hp if x not in (lsvarname + childList)]
    norelationLst = []
    for node in lsOneNode:
        _thisList = []
        for _thisnode in [x[1] for x in lsparentname if x[0] == node]:
            temp = deepcopy(hp[node])
            _newHyperparameter=temp
            _Orig_bounds = [x.bounds for x in temp.bounds]
            _Allvalues = ParamsExt.convert_2levels(_Orig_bounds, [])
            _iterbound = ParamsExt.updatebounds(_thisnode, _Orig_bounds, [])
            # temporary: forget p value.
            # TODO: need a recalculate pvalue function
            _iterbound = _iterbound if isinstance(_iterbound, (tuple, list)) else [_iterbound]
            _selectedvalues = node[1]
            _NewAllvalues = ParamsExt.convert_2levels(_iterbound, [])
            _defaultvalue = temp.default if temp.default in _NewAllvalues else _selectedvalues[0]
            _newHyperparameter = (AlgorithmChoice if isinstance(temp, AlgorithmChoice) else CategoricalParam)(
                _iterbound, temp.var_name, name=temp.name, cutting=temp.cutting, default=_defaultvalue)
            '''_newbounds = []
            _iskeep=True
            for _x in [x for x in item.bounds]:# if len(set(_value).intersection(x.bounds)) > 0]:
                _newvalues=[]
                _temp = deepcopy(_x)
                _newvalues=ParamsExt.updatebounds(_value,_temp.bounds,_newvalues)
                if len(_newvalues)>0:
                    _iskeep=False
                    _temp.bounds = _newvalues
                    _newbounds.append(_temp)
            item.iskeep = True
            item.bounds = _newbounds
            item.allbounds = ParamsExt.convert_2levels([x.bounds for x in _newbounds],[])#[j for i in [x.bounds for x in _newbounds] for j in i]'''
            _thisList.append([_newHyperparameter])
        norelationLst.append(_thisList)
    lsRootNode = [x for x in lsvarname if x not in childList]
    # print(hpa)
    for root in lsRootNode:
        for item in [x for x in lsparentname if x[0] == root]:
            finalA = self._listoutBranches4(item, root, lsvarname, childeffect, lsparentname)
            lsBranches.extend(finalA)
    count, final, MixList = 1, [], []
    for root in lsRootNode:
        tempList = []
        for aBranch in lsBranches:
            in1Lst = False
            for item in aBranch:
                if (item.var_name == root):
                    in1Lst = True
            if (in1Lst == True):
                tempList.append(aBranch)
        MixList.append(tempList)
    MixList = MixList + norelationLst
    # lsRootNode.extend(lsOneNode)
    # Forbidden:
    # 1: Forbidden at module levels: We change the search space
    # 2: Forbidden at node/child/leaves level: check in the sampling function
    listDiffRootForb = OrderedDict()
    if (self._listForbidden != None):
        for id, item in self._listForbidden.forbList.items():
            left = item.left
            l_group, r_group = None, None
            right = item.right
            branch_id = 0
            for modules in MixList:
                for mainBranches in modules:
                    for node in mainBranches:
                        if (node.var_name == left):
                            l_group = branch_id
                        if (node.var_name == right):
                            r_group = branch_id
                branch_id += 1
            if (l_group != r_group):
                item.isdiffRoot = True
                listDiffRootForb[id] = item

    lsFinalSP = []
    FinalSP = OrderedDict()
    MixList_feasible = []
    MixList_feasible_remain = []
    if (len(MixList) > 1):
        final = list(itertools.product(*MixList))
        tobedel = []
        igroup = 0
        for group in final:
            group_new = list(deepcopy(group))
            isDelete = False
            for key, value in listDiffRootForb.items():
                item_left, item_right = None, None
                isBothRoot = False
                if ((value.left in lsRootNode) and (value.right in lsRootNode)):
                    isBothRoot = True
                else:
                    continue
                i = 0
                for module in group_new:
                    # hp_left=[(idx,sp) for (idx,sp) in enumerate(module) if sp.var_name==value.left and len(set(sp.allbounds).intersection(value.leftvalue))>0]
                    # hp_right=[(idx,sp) for (idx,sp) in enumerate(module) if sp.var_name==value.right and len(set(sp.allbounds).intersection(value.rightvalue))>0]
                    hp_left = [(idx, sp) for (idx, sp) in enumerate(module) if sp.var_name == value.left and len(
                        set(sp.allbounds) - set(value.leftvalue)) == 0]
                    hp_right = [(idx, sp) for (idx, sp) in enumerate(module) if sp.var_name == value.right and len(
                        set(sp.allbounds) - set(value.rightvalue)) == 0]
                    if (len(hp_left) > 0):
                        module_left = i
                        index_left, item_left = hp_left[0]
                    if (len(hp_right) > 0):
                        module_right = i
                        index_right, item_right = hp_right[0]
                        # print(index_right)
                    i += 1
                if (item_left != None and item_right != None):
                    sp_bound_left = item_left.allbounds
                    sp_bound_left_remain = tuple(set(sp_bound_left) - set(value.leftvalue))
                    sp_bound_right = item_right.allbounds
                    sp_bound_right = tuple(set(sp_bound_right) - set(value.rightvalue))
                    if (len(sp_bound_right) < 1 and len(sp_bound_left_remain) < 1 and isBothRoot == True):
                        isDelete = True
                    if (len(sp_bound_right) > 0):
                        # item_left.allbounds=sp_bound_left
                        frange = []
                        for bound in item_right.bounds:
                            if (len(set(sp_bound_right).intersection(bound.bounds)) > 0):
                            #if len(np.intersect1d(sp_bound_right,bound.bounds)) > 0:
                                temp_range = bound
                                if isinstance(bound, p_paramrange):
                                    temp_range.p = round(len(sp_bound_right) * (bound.p / len(bound.bounds)), 2)
                                if (isinstance(sp_bound_right, (tuple, list))):
                                    temp_range.bounds = [b for b in sp_bound_right]
                                else:
                                    temp_range.bounds = [sp_bound_right]
                                frange.append(temp_range)
                        item_right.bounds = frange
                        item_right.allbounds = [j for i in [x.bounds for x in frange] for j in i]
                        item_right.default = item_right.default if item_right.default in item_right.allbounds else \
                        item_right.allbounds[0]
                        # item_left=rebuild(item_left)
                        group_new[module_right][index_right] = item_right
                        _del_right_childs = [ke[0] for ke in self._listconditional.conditional.values() if
                                             ke[1] == item_right.var_name
                                             and len(set(value.rightvalue) - set(ke[2])) == 0]
                        if len(_del_right_childs) > 0:
                            lChild_del = []
                            while (len(_del_right_childs) > 0):
                                for right_child in _del_right_childs:
                                    _del_right_childs.extend(
                                        [ke[0] for ke in self._listconditional.conditional.values() if
                                         ke[1] == right_child])
                                    lChild_del.append(right_child)
                                    _del_right_childs.remove(right_child)
                            lIndex_del = [idx for (idx, x) in enumerate(group_new[module_right]) if
                                          x.var_name in lChild_del]
                            for index in sorted(lIndex_del, reverse=True):
                                del group_new[module_right][index]
                    else:
                        if (isDelete == False):
                            right_childs = [ke[0] for ke in self._listconditional.conditional.values() if
                                            ke[1] == item_right.var_name
                                            and len(set(item_right.allbounds) - set(ke[2])) == 0]
                            lIndex_del = [index_right]
                            lChild_del = []
                            # group_new[module_left].pop(index_left)
                            while (len(right_childs) > 0):
                                for right_child in right_childs:
                                    right_childs.extend([ke[0] for ke in self._listconditional.conditional.values() if
                                                         ke[1] == right_child])
                                    lChild_del.append(right_child)
                                    right_childs.remove(right_child)
                            lIndex_del.extend([idx for (idx, x) in enumerate(group_new[module_right]) if
                                               x.var_name in lChild_del])
                            for index in sorted(lIndex_del, reverse=True):
                                del group_new[module_right][index]

            if (isDelete == False):
                final[igroup] = group_new
            else:
                tobedel.append(igroup)
            # MixList_feasible.append(module)
            igroup += 1
        for index in sorted(tobedel, reverse=True):
            del final[index]

        for searchSpace in final:
            space = []
            for group in searchSpace:
                for item in group:
                    # if (item.iskeep == True):
                    # FinalSP[item.var_name[0]] = item
                    space.append(item)
                    # defaults.append(item.default)
            # space.default=defaults
            lsFinalSP.append(space)
            del space
    elif (len(MixList) == 1):
        final = list(MixList)
        for searchSpace in final:
            for group in searchSpace:
                # defaults = []
                space = []
                for item in group:
                    if (item.iskeep == True):
                        FinalSP[item.var_name] = item
                        space.append(item)
                        # defaults.append(item.default)
                # space.default = defaults
                lsFinalSP.append(space)
                del space
    else:
        pass

    return lsFinalSP
def _listoutallnode(self, node, rootname, lsvarname, childeffect,
                               lsparentname, mixlist,feeded):
    temp = deepcopy(self._hyperparameters[node[0]])
    _newHyperparameter=temp
    if (temp.var_name not in feeded):
        feeded.append(temp.var_name)
    if isinstance(temp, (AlgorithmChoice, CategoricalParam)):
        frange = []
        _Orig_bounds=[x.bounds for x in temp.bounds]
        _Allvalues = ParamsExt.convert_2levels(_Orig_bounds, [])
        _thisnode = node[1]
        _iterbound = ParamsExt.updatebounds(_thisnode, _Orig_bounds, [])
        # temporary: forget p value.
        # TODO: need a recalculate pvalue function
        _iterbound = _iterbound if isinstance(_iterbound, (tuple, list)) else [_iterbound]
        _selectedvalues = node[1]
        _NewAllvalues=ParamsExt.convert_2levels(_iterbound, [])
        _defaultvalue= temp.default if temp.default in _NewAllvalues else _selectedvalues[0]

        _newHyperparameter=(AlgorithmChoice if isinstance(temp,AlgorithmChoice) else CategoricalParam)(_iterbound,temp.var_name,name=temp.name,cutting=temp.cutting,default=_defaultvalue)
    _newHyperparameter.iskeep = False
    this_node = [_newHyperparameter]
    if (isinstance(_newHyperparameter, (AlgorithmChoice, CategoricalParam))):
        child_hpa, child_nodes = self._getnodechilds(node, childeffect, lsvarname, lsparentname)
    else:
        child_hpa, child_nodes = [], []
    if (len(child_nodes) > 0):
        for child in child_nodes:
            child_node = self._listoutallnode(child, rootname, lsvarname, childeffect,
                                              lsparentname, mixlist, feeded)
            this_node.extend(child_node)
    if (node[0] == rootname):
        mixlist.extend(this_node)
    return this_node

def _getnodechilds(self,node,childeffect,lsvarname,lsparentname):
    if (isinstance(self._hyperparameters[node[0]], (AlgorithmChoice, CategoricalParam)) == False):
        return [], []
    # child_hpa = [x[1] for x in childeffect if (x[0] == (node[0] + "_" + "".join(str(e) for e in node[1])))]
    #child_hpa = [x[0] for x in self._listconditional.conditional.values() if
    #             (x[1] + "_" + x[2][0]) in (str(node[0]) + "_" + str(e) for e in node[1])]
    child_hpa= [x[0]for x in self._listconditional.conditional.values() if set(x[1]+"_"+str(i) for i in x[2]).intersection(str(node[0]) + "_" + str(e) for e in node[1])]
    n_child = len(child_hpa)
    child_node = []
    if (n_child > 0):
        icount = 0
        # child_node = []
        for child_hpa_i in child_hpa:
            icount += 1
            if (child_hpa_i in lsvarname):
                childlst = [x for x in lsparentname if x[0] == child_hpa_i]
                child_node.extend(childlst)
            else:
                childlst = [child_hpa_i, list(self._hyperparameters[child_hpa_i].allbounds)]
                child_node.append(childlst)
    return child_hpa, child_node
