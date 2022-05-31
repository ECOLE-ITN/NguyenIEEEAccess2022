import copy
from typing import List
import numpy as np
import math
from . import paramrange, p_paramrange, one_paramrange
from . import AlgorithmChoice, HyperParameter, CategoricalParam
__author__ = "Duc Anh Nguyen"
def get_level(algorithmsLst, _i=0):
    _ii = []
    _i += 1
    for x in [j for j in algorithmsLst]:
        # print(x,_i)
        _ii.extend([_i])
        if isinstance(x, list) and (len(x) > 1):
            # _i+=1
            _iii = get_level(x, _i)
            # print('...',x,_iii)
            _ii.extend([_iii])
        else:
            pass
        # print(_ii)
    _i = max(_ii)
    return _i
def convert_2levels(algorithmsLst, new_algorithmsLst=[]):
    for x in [j for j in algorithmsLst]:
        # print(x)
        if isinstance(x, list):
            b = []
            b = convert_2levels(x, b)
            new_algorithmsLst.extend(b)
            '''if any(isinstance(z,list)==True for z in x):                
                return convert_2levels(x,new_a)
            else:
                new_a.append(x)'''
        else:
            new_algorithmsLst.append(x)
    return new_algorithmsLst
def read_tree(algorithmsLst, level=5, _return=[], _i=0):
    _i = _i + 1
    if (_i == level and level != 1) or not any(isinstance(x, list) for x in algorithmsLst):
        # print(_i,"===")
        b = []
        b = convert_2levels(algorithmsLst, b)
        _return.append(b)
        return _return
    elif level == 1:
        b = []
        b = convert_2levels(algorithmsLst, b)
        _return.append(b)
        return _return
    else:
        # print('pass')
        pass
    if isinstance(algorithmsLst, list):
        for x in [j for j in algorithmsLst]:
            # print('---',x,_i)
            if isinstance(x, list):
                b = []
                b = read_tree(x, level, b, _i)
                _return.append(b) if _i == 1 else _return.extend(b)
            else:
                # print('aaaa',x)
                _return.extend([x])
    else:
        # print('bbb',x)
        _return.append(algorithmsLst)
    return _return


def regroup(algorithmsLst, _togroup=3, seed=1):
    rstate = np.random.RandomState(seed)
    #np.random.seed(seed)
    b = []
    b = convert_2levels(algorithmsLst, b)
    if _togroup > len([x for x in b]):
        return False
    if _togroup == 1:
        return [b]
    i = 0
    i = get_level(algorithmsLst, i)
    _lv, _return = 0, []
    for _i in range(1, i + 2):
        _temp = []
        read_tree(algorithmsLst, _i, _return=_temp)
        if _i == 1:
            _numbergroup = 1
        else:
            _numbergroup = len([j for i in _temp for j in i])
        # print(_i,_numbergroup)
        if _numbergroup >= _togroup or _i == i + 1:
            _lv = _i
            _return = _temp
            # else:
            break
    # print(_lv,_return)
    _newgroup, _ungrouped = [], []
    _currentgroup = 0
    for x in _return:
        _temp = [z for z in x if isinstance(z, list)]
        _ungr = [z for z in x if not isinstance(z, list)] if isinstance(x, list) else [x]
        if len(_ungr) > 0:
            _temp.append(_ungr)
        _currentgroup += len(_temp)
        # print(x,_temp)
        _newgroup.append(_temp)
    while _currentgroup != _togroup:
        # print(_currentgroup,_togroup,_newgroup)
        # _p_dict = {i: math.floor((len(v)/len(_grouped)) * 10000) / 10000 for i, v in enumerate(_newgroup)}
        _p_dict, _p_count = dict(), dict()
        for i, v in enumerate(_newgroup):
            _vi = 0
            for _i, _v in {str(i) + '-' + str(_i): len(_v) for _i, _v in enumerate(v)}.items():
                _p_dict[_i] = _v
                _vi += 1  # _v
            _p_count[i] = _vi
        # print(_p_dict,_p_count)
        itemMaxValue = max(_p_dict.items(), key=lambda x: x[1]) if _currentgroup < _togroup else min(_p_dict.items(),
                                                                                                     key=lambda x: x[1])
        # print(_currentgroup,'Maximum Value in Dictionary : ', itemMaxValue,_p_dict)
        listOfKeys = list()
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in _p_dict.items():
            if value == itemMaxValue[1]:
                listOfKeys.append(key)
        if (_currentgroup > _togroup):
            _pvalue = {x: _p_count[int(x.split('-')[0])] for x in listOfKeys}
            _sumP = sum(_pvalue.values())
            _pvalue = [math.floor((x / _sumP) * 100000) / 100000 for i, x in _pvalue.items()]
            _pvalue[-1] += 1 - sum(_pvalue)
            # print('KEY',_pvalue,_pvalue,_p_dict,listOfKeys,_p_count)
            OneKey = rstate.choice(listOfKeys, p=_pvalue, replace=False)
            # print(OneKey,_p_count)
            _OneKeyGroup = OneKey.split('-')[0]
            _numberNeighbours = sum(x for i, x in _p_count.items() if i == int(_OneKeyGroup))
            # print(_numberNeighbours)
            if _numberNeighbours > 1:  # and len(listOfKeys)<2:
                _addItems = []
                _minitem, _minvalue = OneKey, itemMaxValue[1]
                _premin = _minitem.split("-")[0]
                _isOtherBranch = False
                _checkvalue = _minvalue
                _hasneighbour = True if len([x for x in listOfKeys if x.split('-')[0] == _OneKeyGroup]) > 2 else False
                # print(_minitem)
                while len(_addItems) < 1:
                    _addItems = [x for x, value in _p_dict.items() if value == _checkvalue and x != _minitem
                                 and (_premin == x.split("-")[0] if _numberNeighbours > 1 else True)]
                    if len(_addItems) == 0 and _checkvalue > _minvalue and _hasneighbour == False and _p_count[
                        int(_OneKeyGroup)] < 2:
                        _addItems = [x for x, value in _p_dict.items() if
                                     value <= _checkvalue and _premin != x.split("-")[0]]
                        if len(_addItems) > 1:
                            _isOtherBranch = True
                    _checkvalue += 1
                    # print('_checkvalue',_addItems,_checkvalue)
                if _isOtherBranch == False:
                    _selected = rstate.choice(_addItems, size=1)
                    _selected = np.append(_selected, _minitem)
                else:
                    _selected = rstate.choice(_addItems, size=2, replace=False)
                    # _selected=np.append(_selected)
            else:
                # print(_newgroup)
                # print('HERE',OneKey,listOfKeys)
                _premin = OneKey[0].split("-")[0]
                _thisbranch = [x for x in listOfKeys if
                               (x.split("-")[0] == _premin if _numberNeighbours > 1 else True)
                               and x != OneKey]
                if len(_thisbranch) < 1:
                    _thisbranch = [x for x in _p_dict.keys() if
                                   (x.split("-")[0] == _premin if _numberNeighbours > 1 else True)
                                   and x != OneKey]
                # prefer to join the smallest group
                _new_p_count = {x: v for x, v in _p_dict.items() if x in _thisbranch}
                '''#_pvalue={x:_p_count[int(x.split('-')[0])] for x in _new_p_count }
                _sumP=sum(_new_p_count.values())
                _pvalue=[math.floor(((_sumP-x)/_sumP)*100000)/100000 for i,x in _new_p_count.items()]
                _pvalue[-1]+=1-sum(_pvalue)'''
                # print('85', _thisbranch,_pvalue,_p_count,_new_p_count)
                _itemMaxValue = min(_new_p_count.items(), key=lambda x: x[1])
                # print(_currentgroup,'Maximum Value in Dictionary : ', itemMaxValue,_p_dict)
                _listOfKeys = list()
                # Iterate over all the items in dictionary to find keys with max value
                for key, value in _new_p_count.items():
                    if value == _itemMaxValue[1]:
                        _listOfKeys.append(key)
                _secondkey = rstate.choice(_thisbranch, size=1, replace=False) if len(_listOfKeys) > 1 else \
                    _itemMaxValue[0]
                _selected = np.append(OneKey, _secondkey)
                # _selected= np.random.choice(listOfKeys,size=2,replace=False)
            # _selected=-np.sort(_selected)
            # _selected.sort(key = lambda x: x.split("-")[1])
            _selected = sorted(_selected, key=lambda x: x.split("-")[1], reverse=True)
            # _temp=_newgroup.pop(int(_selected[0].split("-")[0]))
            # print(_selected)
            _items = []
            i1, _i1 = _selected[0].split("-")
            for x in _selected:
                i, _i = x.split("-")
                _x = _newgroup[int(i)].pop(int(_i))
                _items.extend(_x)

            # print('-',_currentgroup,_togroup,_items)
            _newgroup[int(i1)].append(_items)
        elif (_currentgroup < _togroup):
            # print('THERE',listOfKeys )
            _selected = rstate.choice(listOfKeys, size=1)
            i, _i = _selected[0].split("-")
            _items = _newgroup[int(i)].pop(int(_i))
            # print(_selected,_items)
            _halfItems = int(np.floor(len(_items) / 2))
            _secondhalf = list(rstate.choice(len(_items), size=_halfItems, replace=False))
            _secondhalf = [_items[i] for i in _secondhalf]
            _firsthalf = sorted(list(set(_items) - set(_secondhalf)), key=lambda x: str(x))
            rstate.shuffle(_firsthalf)
            rstate.shuffle(_secondhalf)
            _newgroup[int(i)].append(_secondhalf)
            _newgroup[int(i)].append(_firsthalf)
            # print('+',_currentgroup,_togroup,_newgroup)
        # _currentgroup=len([j for i in _newgroup for j in i])
        _currentgroup = 0
        for x in _newgroup:
            _currentgroup += len(x)
    # print(_newgroup)
    _return = [j for i in _newgroup for j in i]
    return _return
def updateHyperparameterforBO4ML(hyperparameter: HyperParameter):
    if isinstance(hyperparameter,(AlgorithmChoice,CategoricalParam)):
        _newparam=copy.deepcopy(hyperparameter)
        _bounds=[]
        for x in hyperparameter.bounds:
            _bounds.append(x.bounds)
        _newbounds=[]
        for _item in _bounds:
            if isinstance(_item, list):
                _newbounds.append(convert_2levels(_item, []))
            else:
                _newbounds.extend([_item])
        _newparam=(AlgorithmChoice if isinstance(hyperparameter, AlgorithmChoice) else CategoricalParam)(
                _newbounds, hyperparameter.var_name, name=hyperparameter.name, cutting=hyperparameter.cutting, default=hyperparameter.default)
        return _newparam
    else:
        return hyperparameter
def updatebounds(values, OriBounds, _return:list()=[]):
    _newbounds=[]
    _newbounds=_updatebounds(values, OriBounds, [])
    for x in _newbounds:
        if isinstance(x, list):
            _return.append(convert_2levels(x, []))
        else:
            _return.extend([x])
    return _return
def _updatebounds(values, OriBounds, _return:list()=[]):
    np.random.RandomState(24)
    #print(_return,OriBounds)
    if any(isinstance(x,list) for x in OriBounds):
        for x in OriBounds:
            if isinstance(x,list):
                b=[]
                b=updatebounds(values, x,b)
                if len(b)>0:
                    #_return.append(b)
                    _return.append(b) if len(_return) > 0 else _return.extend(b)
            else:
                #if len(set(values).intersection([x]))>0:
                if x in values:_return.extend([x])
    else:
        _temp_return=sorted(list(set(values).intersection(list(OriBounds))), key=lambda x: str(x))
        _return.extend(_temp_return)
        #_return.extend(list(np.intersect1d(values,list(OriBounds))))
    return _return

def __init__(self, bounds, var_name, name, cutting=None, default=None, hType="C"):
    if isinstance(bounds, (list, tuple,)):
        _thisbound = list()
        _joinbound = list()
        _hasP_param = False
        _allbounds = []
        _thisDef = None
        if (any(isinstance(x, (tuple, list)) for x in bounds) == False):
            _thisbound.append(paramrange(bounds, default=default, hType=hType))
            if (hType == "I"):
                if (len(bounds) == 2):
                    _allbounds = [*range(bounds[0], bounds[1])]
                else:
                    _allbounds = bounds
        else:
            # if(any(isinstance(x,tuple))for x in bounds):
            _sumP = sum([x[1] for x in bounds if isinstance(x, tuple)])
            _hasP_param = True if _sumP > 0 else False
            _itemNoP = len([x for x in bounds if isinstance(x, tuple) == False])
            for bound in bounds:
                if (hType == "I"):
                    if isinstance(bound, tuple):
                        _p = bound[1]
                        if (isinstance(bound[0], list) and len(bound[0]) == 2):
                            _lower = bound[0][0]
                            _upper = bound[0][1]
                            _intbound = [_lower, _upper]
                            _allbounds.extend([*range(_lower, _upper)])
                            bound = (_intbound, _p)
                            if default in _intbound:
                                _thisDef = default
                        else:
                            if isinstance(bound[0], list) == False:
                                bound = ([bound[0]], _p)
                            _allbounds.extend(bound[0])
                            if default in bound[0]:
                                _thisDef = default
                        _thisbound.append(p_paramrange(*bound, default=_thisDef, hType=hType))
                    else:
                        if (isinstance(bound, list) and len(bound) == 2):
                            _lower = bound[0]
                            _upper = bound[1]
                            _allbounds.extend([*range(_lower, _upper)])
                            _intbound = [_lower, _upper]
                            if default in _intbound:
                                _thisDef = default
                            bound = _intbound
                        else:
                            if (isinstance(bound, list) == False):
                                bound = [bound]
                            _allbounds.extend(bound)
                            # _joinbound.append(bound)
                            if default in bound:
                                _thisDef = default
                        if _hasP_param:
                            _p = round(((1 - _sumP) / _itemNoP), 5)
                            _thisbound.append(p_paramrange(bounds=bound, p=_p, default=_thisDef, hType=hType))
                        else:
                            _thisbound.append(paramrange(bounds=bound, default=_thisDef, hType=hType))
                else:
                    if isinstance(bound, tuple):
                        _thisbound.append(p_paramrange(*bound, hType=hType))
                    else:
                        if (isinstance(bound, list) == False):
                            bound = [bound]
                        else:
                            #pass
                            bound=[(x if isinstance(x,list) else [x]) for x in bound] if any([isinstance(x,list) for x in bound]) else bound
                        _thisAllvalues=convert_2levels(bound,[])
                        _allbounds.extend(_thisAllvalues)
                        _thisDef = default if default in _thisAllvalues else _thisAllvalues[0]
                        _p = round(((1 - _sumP) / _itemNoP), 5) if _hasP_param else None
                        _thisItem = p_paramrange(bounds=bound, p=_p, default=_thisDef,
                                                 hType=hType) if _hasP_param else \
                            paramrange(bounds=bound, default=_thisDef, hType=hType)
                        _thisbound.append(_thisItem)

        if (hType in ["C", "A"]):
            _newThisbound = copy.deepcopy(_thisbound)
            for _bound in _newThisbound:
                _2levelbound = []
                _2levelbound = convert_2levels(_bound.bounds, _2levelbound)
                _bound.bounds = _2levelbound if any(isinstance(x, list) for x in _2levelbound) == False \
                    else [j for i in _2levelbound for j in i]
            self.allbounds = convert_2levels([[x.bounds] for x in _newThisbound],[])
        elif hType == "I":
            self.allbounds = _allbounds
        else:
            self.allbounds = [x.bounds for x in _thisbound]
        self.bounds = _thisbound
    else:
        self.allbounds = [bounds]
        self.bounds = [paramrange(bounds, hType)]
        if (default in self.bounds[0].bounds and self.bounds[0].default == None):
            self.bounds[0].default = default
        pass
    self.name = name
    self.var_type = hType
    self.default = default
    self.cutting = cutting
    self.var_name = var_name

def addMutilConditional(self, child: List[HyperParameter] = None, parent: HyperParameter = None, parent_value=None, isRoot=True):
    if isinstance(parent_value, (tuple, list)):
        parent_value = [b for b in parent_value]
    else:
        parent_value = [parent_value]
    _Allvalues = []
    _Allvalues = convert_2levels([x.bounds for x in parent.bounds], _Allvalues)
    if (set(parent_value).issubset(_Allvalues)):
        for achild in child:
            self.addConditional(achild, parent, parent_value, isRoot)
    else:
        raise TypeError("Hyperparameter '%s' is not in range of "
                        "--" %
                        str(parent.var_name))
def addConditional(self, child: HyperParameter = None, parent: HyperParameter = None, parent_value=None, isRoot=None):
    if not isinstance(child, HyperParameter):
        raise TypeError("Hyperparameter '%s' is not an instance of "
                        "DACOpt.SearchSpace" %
                        str(child))
    if not isinstance(parent, HyperParameter):
        raise TypeError("Hyperparameter '%s' is not an instance of "
                        "DACOpt.SearchSpace" %
                        str(parent))
    if isinstance(parent_value, (tuple, list)):
        parent_value = [b for b in parent_value]
    else:
        parent_value = [parent_value]
    _Allvalues=[]
    _Allvalues=convert_2levels([x.bounds for x in parent.bounds],_Allvalues)
    if (set(parent_value).issubset(_Allvalues) == False):
        raise TypeError("Hyperparameter '%s' is not in range of "
                        "--" %
                        str(parent.var_name))
    keyname = str(child.var_name) + '_' + str(parent.var_name)
    # All conditional for impute + bandit
    if (keyname in self.AllConditional.keys()):
        self._updateAllConditional(child, parent, parent_value)
    else:
        self._addAllConditional(child, parent, parent_value)
    # list of conditional for treezation only
    if (isRoot == None):
        if (parent.var_name in [x[0] for i, x in self.conditional.items()]):
            isRoot = False
        else:
            isRoot = True
        if isinstance(parent, AlgorithmChoice):
            isRoot = True
    # if isRoot==True:
    if (keyname in self.conditional.keys()):
        self._updateConditional(child, parent, parent_value, isRoot)
    else:
        self._addConditional(child, parent, parent_value, isRoot)

