from __future__ import absolute_import
from . import BO4ML, Forbidden, \
    ConfigSpace, ConditionalSpace, Extension, rand, tpe, anneal, atpe, Trials
from hyperopt import STATUS_FAIL, STATUS_OK
import numpy as np
import time
__author__ = "Duc Anh Nguyen"
class ObjectiveFunction(object):
    def __init__(self,objFunc, cons:ConditionalSpace, fob: Forbidden, prefix='value', isMinimize=True,isFlatSetting=False):
        self.cons=cons
        self.objFunc=objFunc
        self.FoB=fob
        self.prefix=prefix
        self.isMinimize=isMinimize
        self.isFlat=isFlatSetting

    def call(self,params):
        start=time.time()
        #params = self._getparamflat(params)
        #print(params)
        #_badluck=np.random.choice([0,0],1)
        _params = self._getparamflat(params)
        if self._checkFobidden(_params):
            #print('invalid')
            return {'loss': 1 if self.isMinimize else 0, 'status': STATUS_FAIL, 'runtime': time.time() - start, 'msg': "INVALID PARAMS"}
        activeLst=self.getActive(_params)
        for x in [x for x in _params.keys() if x not in activeLst]:
            #pass
            print('+++++++++++++++++++++++++++++++++++++++++++',x,'++++++++++++++++++++++++++++++++++')
        return self.objFunc(_params if self.isFlat else params)
    def _checkFobidden(self, x_dict):
        _forbidden = self.FoB
        isFOB = False

        if (_forbidden != None):
            for fname, fvalue in _forbidden.forbList.items():
                #print(fname, fvalue.leftvalue, fvalue.rightvalue)
                hp_left = [(key, value) for (key, value) in x_dict.items() if
                           key == fvalue.left and len(set([value]).intersection(fvalue.leftvalue)) > 0]
                hp_right = [(key, value) for (key, value) in x_dict.items() if
                            key == fvalue.right and len(set([value]).intersection(fvalue.rightvalue)) > 0]
                hp_add1, hp_add2 = [], []
                if (fvalue.ladd1 != None):
                    hp_add1 = [(key, value) for (key, value) in x_dict.items() if
                               key == fvalue.ladd1 and len(set([value]).intersection(fvalue.ladd1value)) > 0]
                if (fvalue.ladd2 != None):
                    hp_add2 = [(key, value) for (key, value) in x_dict.items() if
                               key == fvalue.ladd2 and len(set([value]).intersection(fvalue.ladd2value)) > 0]
                if (fvalue.ladd1 != None and fvalue.ladd2 != None):
                    if (len(hp_left) > 0 and len(hp_right) > 0 and len(hp_add1) > 0 and len(hp_add2) > 0):
                        isFOB = True
                elif (fvalue.ladd1 != None):
                    if (len(hp_left) > 0 and len(hp_right) > 0 and len(hp_add1) > 0):
                        isFOB = True
                else:
                    if (len(hp_left) > 0 and len(hp_right) > 0):
                        isFOB = True
        return isFOB

    '''def convertparams(self, params):
        # print(params)
        #params=self._getparamflat(params)
        ActiveLst = self.getActive(params)
        # print(ActiveLst)
        listvalues = dict()
        for x in ActiveLst:
            key = self.decode(x)
            value = params[x]
            if (isinstance(value, int)):
                value = int(value)
            elif (isinstance(value, float)):
                value = float(value)
            # print(type(value))
            listvalues[key] = value
        return listvalues'''

    def _getparamflat(self, params, parent=None):
        _return = dict()
        for i, x in params.items():
            if isinstance(x, dict):
                _x = self._getparamflat(x, i)
                _return.update(_x)
            else:
                _return[parent if i == self.prefix else i] = x
        return _return

    def paramFormat(self, params):
        # params = {k: params[k] for k in params if params[k]}
        for k, v in params.items():
            if (isinstance(v, dict)):
                params[k] = self.paramFormat(v)
            if (v == 'True' or v == 'true'):
                params[k] = True
            elif (v == 'False' or v == 'false'):
                params[k] = False
            elif (v == 'None'):
                params[k] = None
        # params=_getparamflat(params,None)
        return params

    def getActive(self, params):
        params = self._getparamflat(params)
        lsParentName, childList, lsFinalSP, ActiveLst = [], [], [], []
        for i, item in self.cons.AllConditional.items():
            if ([item[1], item[2], item[0]] not in lsParentName):
                lsParentName.append([item[1], item[2], item[0]])
            if (item[0] not in childList):
                childList.append(item[0])
        lsRootNode = [x for x in params.keys() if x not in childList]
        for root in lsRootNode:
            rootvalue = params[root]
            ActiveLst.append(root)
            # print(root,rootvalue)
            for node, value in [(x[2], x[1]) for x in lsParentName if x[0] == root and rootvalue in x[1]]:
                value = params[node]
                ActiveLst.append(node)
                nodeChilds = [(x[2], x[1]) for x in lsParentName if x[0] == node and value in x[1]]
                while (len(nodeChilds) > 0):
                    childofChild = []
                    for idx, child in enumerate(nodeChilds):
                        childvalue = params[child[0]]
                        # print("--",child[0],childvalue)
                        childofChild.extend(
                            [(x[2], x[1]) for x in lsParentName if x[0] == child[0] and childvalue in x[1]])
                        ActiveLst.append(child[0])
                        del nodeChilds[idx]
                    if (len(childofChild) > 0):
                        nodeChilds = childofChild
        return ActiveLst