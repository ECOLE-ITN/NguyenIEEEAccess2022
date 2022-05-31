from sklearn.model_selection import cross_val_score 
from autosklearn.pipeline.classification import SimpleClassificationPipeline
#from autosklearn.pipeline import util as putil
import autosklearn.util.pipeline as PP
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from hpolib.abstract_benchmark import AbstractBenchmark
from typing import Generator, Optional
import openml, os
import scipy.sparse
from sklearn.model_selection import train_test_split, KFold
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CONs
import ConfigSpace.forbidden as forb
import ConfigSpace.configuration_space as conf
from dacopt import ConfigSpace, ConditionalSpace, AlgorithmChoice, IntegerParam, FloatParam, CategoricalParam, Forbidden
import math
import numpy as np
import pandas as pd
class OpenMLDataManager():
    #https://github.com/Ennosigaeon/automl_benchmark/
    def __init__(self, openml_task_id: int, rng=None):
        self.X = None
        self.y = None
        self.categorical = None
        self.names = None
        self.folds = []

        self.save_to = os.path.expanduser('./DATA/OpenML')
        self.task_id = openml_task_id

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if not os.path.isdir(self.save_to):
            #logger.debug('Create directory {}'.format(self.save_to))
            os.makedirs(self.save_to)

        openml.config.apikey = 'bc304d5d0575357765f084af447c02ff'
        openml.config.set_cache_directory(self.save_to)

    def load(self, shuffle: bool = False) -> 'OpenMLDataManager':
        '''
        Loads dataset from OpenML in _config.data_directory.
        Downloads data if necessary.
        Returns
        -------
        X_train: np.array
        y_train: np.array
        X_test: np.array
        y_test: np.array
        '''
        task = openml.tasks.get_task(self.task_id)
        dataset = openml.datasets.get_dataset(dataset_id=task.dataset_id)
        X, y, categorical, self.names = dataset.get_data(
            target=dataset.default_target_attribute
        )

        for name, cat in zip(self.names, categorical):
            if cat:
                enc = LabelEncoder()
                missing = np.any(pd.isna(X[name]))

                missing_vec = pd.isna(X[name])

                x_tmp = X[name].cat.add_categories(['<MISSING>']).fillna('<MISSING>')
                X[name] = enc.fit_transform(x_tmp)

                if missing:
                    idx = enc.transform(['<MISSING>'])[0]
                    X[name][X[name] == idx] = np.nan
                    assert pd.isna(X[name]).equals(missing_vec)

        X = X.values
        y = y.values.__array__()
        self.y = LabelEncoder().fit_transform(y)
        self.X = X.astype(np.float64)

        if shuffle:
            shuffle = self.rng.permutation(X.shape[0])
            self.X = self.X[shuffle[:]]
            self.y = self.y[shuffle[:]]

        self.categorical = categorical
        self.num_classes = len(np.unique(y))        
        self.isSpare = scipy.sparse.issparse(X)
        self.dataset_id=task.dataset_id
        self.variable_types = ['categorical' if c else 'numerical' for c in categorical]
        self.variable_types={i:v for i,v in zip(self.names,self.variable_types)}
        return self
class OpenMLHoldoutDataManager(OpenMLDataManager):

    def load(self, test_size: float = 0.3) -> 'OpenMLHoldoutDataManager':
        super().load()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size,random_state=self.rng)
        ls = [X_train, y_train, X_test, y_test]
        self.folds.append(ls)
        return self


class OpenMLCVDataManager(OpenMLDataManager):

    def load(self, n_splits: int = 4) -> 'OpenMLCVDataManager':
        super().load()
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            ls = [X_train, y_train, X_test, y_test]
            self.folds.append(ls)
        return self

class ExpSupport():
    def __init__(self,dataset,include, exclude,BO_type,task,isSpare):
        self.BO_type=BO_type
        self.dataset=dataset
        self.isSpare=isSpare
        self.cs = PP.get_configuration_space(
            info={'task': task, 'is_sparse': isSpare},
            include_estimators=include.get('classifier'),
            include_preprocessors=include.get('preprocessor'))

        self.nameconvert=dict()
        i=0
        for c in self.cs.get_hyperparameters():
            i=i+1
            #print(c)
            name=c.name
            name2="p"+str(i)
            self.nameconvert[name]=name2
        self.convert2BO4ml()
    def convert2BO4ml(self):
        search_space = ConfigSpace() 
        nforb = Forbidden() 
        #runId = NominalSpace([str(runId)], 'runId')
        listParamName=[]
        for ca in self.cs.get_hyperparameters():
            pName,pType,pValue,pdefault, scale = self.getParamvalues(ca)
            pName=self.nameconvert[pName]
            listParamName.append(pName)
            #scale=",scale='"+scale+"'" if scale!= None else ""
            scale=""
            txt = pName +"= "+pType+"("+str(pValue)+",'"+pName+"',default="+str(pdefault)+scale+")"
            addtosp="search_space._add_singleparameter("+pName+")"    
            #print(txt)
            #print(addtosp)
            exec(txt)
            exec(addtosp)
        con = ConditionalSpace('conditional') 
        for cons in self.cs.get_conditions():
            father, child, fvalue = self.getConditional(cons)
            father=self.nameconvert[father]
            child=self.nameconvert[child]
            #txt = "con.addConditional()"
            #print(father, child, fvalue)
            txt="con.addConditional("+child+","+father+","+fvalue+")"
            #print(txt)
            exec(txt)
        myforb = Forbidden()
        for f in self.cs.get_forbiddens():
            temp=''
            #print(f)
            for f1 in f.components:
                name=f1.hyperparameter.name
                name2=self.nameconvert[name]
                #ftype=type(f1)
                fcon = ","
                if (isinstance(f1,forb.ForbiddenInClause)):
                    value= f1.values
                    value=[x for x in value]
                elif(isinstance(f1,forb.ForbiddenEqualsClause)):
                    value= f1.value
                    if (value.isnumeric()):
                        value=int(value)
                    else:
                        try:
                            value=float(value)
                        except:
                            value="'"+str(value)+"'"            
                else:
                    fcon=" "
                temp=temp+""+str(name2)+fcon+str(value).replace("'",'"')+","
            temp=temp[:-1]
            txt= "myforb.addForbidden("+temp+")"
            #print(txt)
            exec(txt)
            #finallist=finallist+temp+'\n'
            self.search_space=search_space
            self.myforb=myforb
            self.con=con

    def decode(self,INvalue):
        for key, value in self.nameconvert.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if (value == INvalue):
                return key
    def convertsmacparams(self, params):    
        ActiveLst = self.getActive(params)
        for x in [x for x in params.keys() if x not in ActiveLst]:
            params[x]=None
        smacparams=self.conf.Configuration
        listvalues=dict()
        for key,value in params.items():    
            key=self.decode(key)
            if (isinstance(value,int)):
                value=int(value)
            elif(isinstance(value, float)):
                value=float(value)

            #print(type(value))
            listvalues[key]=value
        #print(listvalues)
        smacparams=self.smacparams(listvalues)

        return smacparams

    def getActive(self,params):
        params=self._getparamflat(params)
        lsParentName, childList, lsFinalSP, ActiveLst = [], [], [], []
        for i, item in self.con.AllConditional.items():
            if ([item[1], item[2], item[0]] not in lsParentName):
                lsParentName.append([item[1], item[2], item[0]])
            if (item[0] not in childList):
                childList.append(item[0])
        lsRootNode = [x for x in params.keys() if x not in childList]
        for root in lsRootNode:
            rootvalue = params[root]
            ActiveLst.append(root)
            #print(root,rootvalue)
            for node, value in [(x[2], x[1]) for x in lsParentName if x[0] == root and rootvalue in x[1]]:
                value=params[node]
                ActiveLst.append(node)
                nodeChilds = [(x[2], x[1]) for x in lsParentName if x[0] == node and value in x[1]]
                while (len(nodeChilds) > 0):
                    childofChild=[]
                    for idx, child in enumerate(nodeChilds):
                        childvalue= params[child[0]]
                        #print("--",child[0],childvalue)
                        childofChild.extend([(x[2], x[1]) for x in lsParentName if x[0] == child[0] and childvalue in x[1]])
                        ActiveLst.append(child[0])
                        del nodeChilds[idx]
                    if(len(childofChild)>0):
                        nodeChilds=childofChild
        return ActiveLst
    
    def getParamvalues(self,c):
        name=c.name
        default= c.default_value
        scale=None
        if (isinstance(c,CSH.CategoricalHyperparameter)):
            if (self.BO_type in ['Bandit','full']):
                choices=[]
                if (name=='classifier:__choice__') :
                    choices=[
                        #1.1.Linear classification
                        ['liblinear_svc','sgd',['passive_aggressive']],                        
                        #1.4Support Vector Machines#1.6. Nearest Neighbors
                        ['libsvm_svc','k_nearest_neighbors','mlp'],  
                        #1.2. Linear and Quadratic Discriminant Analysis# 1.9. Naive Bayes
                        [['qda','lda'],['gaussian_nb','bernoulli_nb', 'multinomial_nb']],                        
                        #1.11 Ensemble methods: GradientBoostingClassifier#1.10. Decision Trees
                        ['adaboost','gradient_boosting'], 
                        [['random_forest'],['decision_tree','extra_trees']]                                             
                        #1.17. Neural network models(Multi-layer Perceptron, 
                        ]                        
                elif(name=='feature_preprocessor:__choice__'):
                    if(self.isSpare):
                        choices=[
                            #sklearn.ensemble#scipy
                            [['extra_trees_preproc_for_classification','random_trees_embedding'],'densifier'],
                            #sklearn.preprocessing.PolynomialFeatures#no
                            [[ 'kernel_pca','polynomial'],['no_preprocessing']],
                            #sklearn.kernel_approximation.RBFSampler-Nystroem
                            #Linear: sklearn.svm#sklearn.decomposition
                            [['kitchen_sinks','nystroem_sampler'],'liblinear_svc_preprocessor','truncatedSVD'],                             
                            #sklearn.feature_selection.SelectPercentile, GenericUnivariateSelect
                            ['select_percentile_classification', 'select_rates_classification']
                            ]
                    else:
                        choices=[
                            #sklearn.ensemble#sklearn.cluster
                            [['extra_trees_preproc_for_classification', 'random_trees_embedding'], 'feature_agglomeration'],
                            #sklearn.decomposition #sklearn.preprocessing.PolynomialFeatures#No
                            [['fast_ica', 'kernel_pca','polynomial'],['no_preprocessing']],  
                            #sklearn.kernel_approximation.RBFSampler-Nystroem#sklearn.svm
                            [['kitchen_sinks', 'nystroem_sampler'],'liblinear_svc_preprocessor', 'pca'],                
                            #sklearn.feature_selection.SelectPercentile, GenericUnivariateSelect
                            ['select_percentile_classification', 'select_rates_classification']
                            ]
                elif(name=='data_preprocessing:numerical_transformer:rescaling:__choice__'):
                    if (self.isSpare):
                        choices=[
                            #6.3.1. Standardization, or mean removal and variance scaling
                            ['standardize','robust_scaler'], 
                            #6.3.2. Non-linear transformation
                            'quantile_transformer',
                            #6.3.3. Normalization
                            'normalize',
                            #None
                            'none']
                    else:
                        choices=[
                            #6.3.1. Standardization, or mean removal and variance scaling
                            ['standardize','minmax','robust_scaler'],
                            #6.3.2. Non-linear transformation
                            ['quantile_transformer','power_transformer'],
                            #6.3.3. Normalization
                            'normalize',
                            #None   
                            'none']
                else:
                    choices = list(c.choices)
                    #print(choices)
            else:
                choices = list(c.choices)
            ctype= "AlgorithmChoice" if '__choice__' in name else "CategoricalParam"
            #ctype= "AlgorithmChoice" if name in ('classifier:__choice__','feature_preprocessor:__choice__','data_preprocessing:numerical_transformer:rescaling:__choice__') else "CategoricalParam"
            if(isinstance(choices[0],int)):
                default=str(c.default_value)
            else:
                default= "'"+str(c.default_value)+"'"
        elif(isinstance(c,CSH.OrdinalHyperparameter)):
            choices=[c.sequence]
            ctype="IntegerParam"

        elif(isinstance(c,CSH.UniformIntegerHyperparameter)):
            choices=[c.lower, c.upper+1]
            ctype='IntegerParam'

        elif(isinstance(c,CSH.UniformFloatHyperparameter)):
            choices=[c.lower, c.upper]
            scale="loguniform" if c.log==True else None
            ctype='FloatParam'

        elif(isinstance(c,CSH.Constant)):
            ctype= "AlgorithmChoice" if '__choice__' in name else "CategoricalParam"
            choices=c.value
            if(isinstance(c.value,str)):            
                choices="['"+str(choices)+"']"
                default= "'"+str(c.default_value)+"'"
            elif(isinstance(c.value,int)):
                choices="["+str(choices)+"]"
                default= str(c.default_value)
            elif(isinstance(c.value,float)):
                choices="["+str(choices)+"]"
                default= str(c.default_value)
        else:
            print("========",c)
            choices='Unexpected'
        return name, ctype, choices, default, scale
    def getConditional(self,cons):
        father=cons.parent.name
        child=cons.child.name
        fvalue=cons.value
        fvalue=[fvalue] if isinstance(fvalue,str) else fvalue
        if (isinstance(cons,CONs.NotEqualsCondition)):
            _father_hpx=self.cs.get_hyperparameter(father)
            pName,pType,pValue,pdefault, scale = self.getParamvalues(_father_hpx)
            fvalue=str(list(set(pValue)-set(fvalue)))    
        fvalue=str(fvalue)
        return father, child, fvalue
    def getConditional2(self,cons):
        father=cons.parent.name
        child=cons.child.name
        fvalue=cons.value
        fvalue=[fvalue] if isinstance(fvalue,str) else fvalue
        if(isinstance(cons,CONs.InCondition)):  
            fvalue=str(fvalue)
        elif(isinstance(cons,CONs.NotEqualsCondition)):
            _father_hpx=self.cs.get_hyperparameter(father)
            pName,pType,pValue,pdefault, scale = self.getParamvalues(_father_hpx)
            fvalue=str(list(set(pValue)-set(fvalue)))
        else:
            '''print(fvalue,type(fvalue))
            fvalue="['"+str(fvalue)+"']"
            print(fvalue,type(fvalue))'''
            #print(fvalue,type(fvalue))
            pass
        fvalue=str(fvalue)
        return father, child, fvalue
    # In[11]:
    def _checkFobidden(self,x_dict):
        _forbidden=self.myforb
        isFOB=False
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

        #print(hp_left, hp_right)
        return isFOB
    def convertparams(self,params):   
        #print(params)
        #params=_getparamflat(params)
        ActiveLst = self.getActive(params)
        #print(ActiveLst)
        listvalues=dict()
        for x in ActiveLst:
            key=self.decode(x)
            value=params[x]
            if (isinstance(value,int)):
                value=int(value)
            elif(isinstance(value, float)):
                value=float(value)
            #print(type(value))
            listvalues[key]=value
        return listvalues
    def _getparamflat(self,params, parent=None, prefix="value"):
        _return = dict()
        for i, x in params.items():
            if isinstance(x, dict):
                _x = self._getparamflat(x, i,prefix)
                _return.update(_x)
            else:
                _return[parent if i == prefix else i] = x
        return _return
    def paramFormat(self,params):
        #params = {k: params[k] for k in params if params[k]}
        for k, v in params.items():
            if (isinstance(v, dict)):
                params[k] = self.paramFormat(v)
            if (v == 'True' or v == 'true'):
                params[k] = True
            elif (v == 'False' or v == 'false'):
                params[k] = False
            elif (v == 'None'):
                params[k] = None
        #params=_getparamflat(params,None)
        return params