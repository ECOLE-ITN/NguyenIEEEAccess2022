from dacopt import DACOpt, ConfigSpace, ConditionalSpace, AlgorithmChoice, IntegerParam, FloatParam, CategoricalParam
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#from DACOpt import paramrange,p_paramrange,one_paramrange

randomstate=9
if 1==1:
    search_space = ConfigSpace()
    con = ConditionalSpace("test")
    random_state = CategoricalParam(randomstate, 'random_state')
    alg_namestr = AlgorithmChoice(['SVM', 'RF', 'KNN', 'DTC', 'LR'], 'classifier', default='SVM')
    search_space.add_multiparameter([random_state, alg_namestr])
    # SVM
    probability = CategoricalParam([True, False], 'probability')
    C = FloatParam([0.03125, 200], 'C')
    kernel = CategoricalParam(['linear', 'rbf', 'poly', 'sigmoid'], 'kernel', default='linear')
    degree = IntegerParam([2, 5], 'degree')
    gamma = CategoricalParam([['auto', 'scale'], 'value'], 'gamma', default='auto')
    gamma_value = FloatParam([3.1E-05, 8], 'gamma_value')
    coef0 = FloatParam([-1, 1], 'coef0')
    shrinking = CategoricalParam([True, False], 'shrinking')
    tol_svm = FloatParam([1e-05, 1e-01], 'tol')
    search_space.add_multiparameter([probability, C, kernel, degree, gamma, gamma_value, coef0, shrinking, tol_svm])
    con.addMutilConditional([probability, C, kernel, degree, gamma, gamma_value, coef0, shrinking, tol_svm],
                            alg_namestr, 'SVM')
    # con.addConditional(gamma_value, gamma,'value')
    ##RF
    n_estimators = IntegerParam([1, 150], 'n_estimators')
    criterion = CategoricalParam(['gini', 'entropy'], 'criterion')
    max_features_RF = CategoricalParam([1, 'sqrt', 'log2', None], 'max_features')
    min_samples_split = IntegerParam([2, 20], 'min_samples_split')
    min_samples_leaf = IntegerParam([1, 20], 'min_samples_leaf')
    bootstrap = CategoricalParam([True, False], 'bootstrap')
    class_weight = CategoricalParam([['balanced', 'balanced_subsample'], None], 'class_weight')
    search_space.add_multiparameter(
        [n_estimators, criterion, max_features_RF, min_samples_split, min_samples_leaf, bootstrap, class_weight])
    con.addMutilConditional([n_estimators, criterion, max_features_RF, min_samples_split,
                             min_samples_leaf, bootstrap, class_weight], alg_namestr, 'RF')
    ###KNN
    n_neighbors_knn = IntegerParam([1, 51], 'n_neighbors_knn')
    weights = CategoricalParam(['uniform', 'distance'], 'weights')
    algorithm = CategoricalParam(['auto', 'ball_tree', 'kd_tree', 'brute'], 'algorithm')
    p = IntegerParam([0, 20], 'p_value')
    search_space.add_multiparameter([n_neighbors_knn, weights, algorithm, p])
    con.addMutilConditional([n_neighbors_knn, weights, algorithm, p], alg_namestr, 'KNN')
    ####DTC
    criterion_dtc = CategoricalParam(['gini', 'entropy'], 'criterion_dtc')
    max_features_dtc = CategoricalParam([1, 'sqrt', 'log2', None], 'max_features_dtc')
    max_depth = IntegerParam([2, 20], 'max_depth_dtc')
    min_samples_split_dtc = IntegerParam([2, 20], 'min_samples_split_dtc')
    min_samples_leaf_dtc = IntegerParam([1, 20], 'min_samples_leaf_dtc')
    # search_space.add_multiparameter([max_depth])
    # con.addMutilConditional([criterion,max_features_RF,min_samples_split,min_samples_leaf,max_depth],alg_namestr,"DTC")
    search_space.add_multiparameter(
        [criterion_dtc, max_features_dtc, max_depth, min_samples_split_dtc, min_samples_leaf_dtc])
    con.addMutilConditional([criterion_dtc, max_features_dtc, max_depth, min_samples_split_dtc, min_samples_leaf_dtc],
                            alg_namestr, "DTC")
    #####LR
    C_lr = FloatParam([0.03125, 100], 'C_LR')
    penalty_solver = CategoricalParam([['l1+liblinear', 'l2+liblinear'],
                                       ['l1+saga', 'l2+saga', 'elasticnet+saga', 'none+saga'], ['l2+sag', 'none+sag'],
                                       ['l2+newton-cg', 'none+newton-cg'], ['l2+lbfgs', 'none+lbfgs']],
                                      'penalty_solver')
    tol_lr = FloatParam([1e-05, 1e-01], 'tol_lr')
    l1_ratio = FloatParam([1e-09, 1], 'l1_ratio')
    search_space.add_multiparameter([C_lr, penalty_solver, tol_lr, l1_ratio])
    con.addMutilConditional([C_lr, penalty_solver, tol_lr, l1_ratio], alg_namestr, 'LR')
    smo_type = AlgorithmChoice([['NO'], ['SMOTEENN', 'SMOTETomek'],
                                [['SMOTE', 'BorderlineSMOTE', 'KMeansSMOTE', 'SMOTENC', 'SVMSMOTE'],
                                 'RandomOverSampler', 'ADASYN'],
                                [['ClusterCentroids'], ['NearMiss', 'RandomUnderSampler'],
                                 [['TomekLinks', 'OneSidedSelection'], 'InstanceHardnessThreshold',
                                  ['EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 'AllKNN'],
                                  ['CondensedNearestNeighbour', 'NeighbourhoodCleaningRule']]]], 'resampler')

    search_space._add_singleparameter(smo_type)
    k_neighbors_SMOTE = IntegerParam([1, 10], 'k_neighbors_SMOTE')
    k_neighbors_Borderline = IntegerParam([1, 10], 'k_neighbors_Borderline')
    m_neighbors_Borderline = IntegerParam([1, 10], 'm_neighbors_Borderline')
    kind = CategoricalParam(['borderline-1', 'borderline-2'], 'kind')
    categorical_features = CategoricalParam([True], 'categorical_features')
    k_neighbors_SMOTENC = IntegerParam([1, 10], 'k_neighbors_SMOTENC')
    k_neighbors_SVMSMOTE = IntegerParam([1, 10], 'k_neighbors_SVMSMOTE')
    m_neighbors_SVMSMOTE = IntegerParam([1, 10], 'm_neighbors_SVMSMOTE')
    out_step = FloatParam([0, 1], 'out_step')
    k_neighbors_KMeansSMOTE = IntegerParam([1, 10], 'k_neighbors_KMeansSMOTE')
    cluster_balance_threshold = FloatParam([1e-2, 1], 'cluster_balance_threshold')
    n_neighbors_OVER = IntegerParam([1, 10], 'n_neighbors_OVER')
    search_space.add_multiparameter([k_neighbors_SMOTE, k_neighbors_Borderline, m_neighbors_Borderline, kind,
                                     categorical_features, k_neighbors_SMOTENC,
                                     k_neighbors_SVMSMOTE, m_neighbors_SVMSMOTE, out_step,
                                     k_neighbors_KMeansSMOTE, cluster_balance_threshold, n_neighbors_OVER])
    con.addConditional(k_neighbors_SMOTE, smo_type, 'SMOTE')
    con.addMutilConditional([k_neighbors_Borderline, m_neighbors_Borderline, kind], smo_type, 'BorderlineSMOTE')
    con.addMutilConditional([categorical_features, k_neighbors_SMOTENC, ], smo_type, 'SMOTENC')
    con.addMutilConditional([k_neighbors_SVMSMOTE, m_neighbors_SVMSMOTE, out_step], smo_type, 'SVMSMOTE')
    con.addMutilConditional([k_neighbors_KMeansSMOTE, cluster_balance_threshold], smo_type, 'KMeansSMOTE')
    con.addConditional(n_neighbors_OVER, smo_type, 'ADASYN')
    n_neighbors_UNDER50 = IntegerParam([1, 50], 'n_neighbors_CNN')
    n_seeds_S = IntegerParam([1, 50], 'n_seeds_S_CNN')
    n_neighbors_UNDER1 = IntegerParam([1, 20], 'n_neighbors_UNDER1')
    kind_sel1 = CategoricalParam(['all', 'mode'], 'kind_sel1')
    n_neighbors_UNDER2 = IntegerParam([1, 20], 'n_neighbors_UNDER2')
    kind_sel2 = CategoricalParam(['all', 'mode'], 'kind_sel2')
    n_neighbors_UNDER3 = IntegerParam([1, 20], 'n_neighbors_UNDER3')
    kind_sel3 = CategoricalParam(['all', 'mode'], 'kind_sel3')
    allow_minority = CategoricalParam([True, False], 'allow_minority')
    estimator_IHT = CategoricalParam(['knn', 'decision-tree', 'adaboost', 'gradient-boosting', 'linear-svm', None],
                                     'estimator_IHT')
    cv_under = IntegerParam([2, 20], 'cv')
    version = CategoricalParam([1, 2, 3], 'version')
    n_neighbors_UNDER4 = IntegerParam([1, 20], 'n_neighbors_UNDER4')
    n_neighbors_ver3 = IntegerParam([1, 20], 'n_neighbors_ver3')
    n_neighbors_UNDER5 = IntegerParam([1, 20], 'n_neighbors_UNDER5')
    threshold_cleaning_NCR = FloatParam([0, 1], 'threshold_cleaning')
    n_neighbors_UNDER6 = IntegerParam([1, 20], 'n_neighbors_UNDER6')
    n_seeds_S_under = IntegerParam([1, 20], 'n_seeds_S')
    replacement = CategoricalParam([True, False], 'replacement')
    estimator_CL = CategoricalParam(['KMeans', 'MiniBatchKMeans'], 'estimator')
    voting_CL = CategoricalParam(['hard', 'soft'], 'voting')
    search_space.add_multiparameter([n_neighbors_UNDER50, n_seeds_S, n_neighbors_UNDER1, kind_sel1,
                                     n_neighbors_UNDER2, kind_sel2, n_neighbors_UNDER3, kind_sel3,
                                     allow_minority, estimator_IHT, cv_under, version, n_neighbors_UNDER4,
                                     n_neighbors_ver3,
                                     n_neighbors_UNDER5, threshold_cleaning_NCR, n_neighbors_UNDER6, n_seeds_S_under,
                                     replacement, estimator_CL, voting_CL
                                     ])
    con.addMutilConditional([n_neighbors_UNDER50, n_seeds_S], smo_type, 'CondensedNearestNeighbour')
    con.addMutilConditional([n_neighbors_UNDER1, kind_sel1], smo_type, 'EditedNearestNeighbours')
    con.addMutilConditional([n_neighbors_UNDER2, kind_sel2], smo_type, 'RepeatedEditedNearestNeighbours')
    con.addMutilConditional([n_neighbors_UNDER3, kind_sel3, allow_minority], smo_type, 'AllKNN')
    con.addMutilConditional([estimator_IHT, cv_under], smo_type, 'InstanceHardnessThreshold')
    con.addMutilConditional([version, n_neighbors_UNDER4, n_neighbors_ver3], smo_type, 'NearMiss')
    con.addMutilConditional([n_neighbors_UNDER5, threshold_cleaning_NCR], smo_type, 'NeighbourhoodCleaningRule')
    con.addMutilConditional([n_neighbors_UNDER6, n_seeds_S_under], smo_type, 'OneSidedSelection')
    con.addConditional(replacement, smo_type, 'RandomUnderSampler')
    con.addMutilConditional([estimator_CL, voting_CL], smo_type, 'ClusterCentroids')

from dacopt import OrginalToHyperopt

#HPOsearchspace = OrginalToHyperopt(search_space._hyperparameters, con, "name")
iris = datasets.load_iris()
X = iris.data
y = iris.target

resampler_group={'NO':'NO','NONE':'NONE','SMOTE':'OVER','BorderlineSMOTE':'OVER','SMOTENC':'OVER','SVMSMOTE':'OVER','KMeansSMOTE':'OVER'
                 ,'ADASYN':'OVER','RandomOverSampler':'OVER',
                 'SMOTEENN':'COMBINE','SMOTETomek':'COMBINE','A':'A','B':'B',
                 'CondensedNearestNeighbour':'UNDER','EditedNearestNeighbours':'UNDER',
                 'RepeatedEditedNearestNeighbours':'UNDER','AllKNN':'UNDER',
                 'InstanceHardnessThreshold':'UNDER','NearMiss':'UNDER',
                            'NeighbourhoodCleaningRule':'UNDER','OneSidedSelection':'UNDER','RandomUnderSampler':'UNDER',
                            'TomekLinks':'UNDER','ClusterCentroids':'UNDER'}
rstate = np.random.RandomState(9)
def new_obj(params):
    #print(params)
    global resampler_group,i,rstate
    i=i+1
    '''Anh=params['Anh']['name']
    time.sleep(0.01)
    params_C = params['classifier']
    #time.sleep(np.random.choice([0,0.5,1,3]))
    classifier = params_C.pop('name')
    #if classifier=='SVM':
    #    print(params_C['gamma'])
    p_sub_params = params.pop('resampler')
    p_sub_type = p_sub_params.pop('name') if isinstance(p_sub_params,dict) else p_sub_params
    sampler = resampler_group[p_sub_type]
    #print(classifier,p_sub_type,sampler)
    _result = np.random.uniform(0, 0.5) if Anh=='ANH' else np.random.uniform(0.1,1)'''
    _result =np.random.uniform(0, 1)
    #print(i,classifier,sampler,p_sub_type,_result)
    #if i>248:
    #print(i,_result)
    return _result
def obj_func(params):
    print(params)
    params = {k: params[k] for k in params if params[k]}
    # print(params)
    seed = params['random_state']
    params = params['classifier']
    classifier = params.pop('name')
    # print(params)
    clf = SVC()
    if (classifier == 'SVM'):
        clf = SVC(**params, random_state=seed)
    elif (classifier == 'RF'):
        clf = RandomForestClassifier(**params, random_state=seed)
    #cv = StratifiedKFold(y, n_splits=2,random_state=seed)
    mean = cross_val_score(clf, X, y).mean()
    loss = 1 - mean
    #print (mean)
    return loss

i=0
from hyperopt import Trials
import time
thistrial=Trials()
opt = DACOpt(search_space, new_obj, conditional=con, hpo_prefix="name", isDaC=True,
                HPOopitmizer='hpo', random_seed=1, max_threads=2
                ,eta=2,hpo_trials=thistrial,compare_strategy='highest',
                max_eval=250,hpo_algo='TPE',show_message=True,
            number_candidates=10, timeout= None#,n_init_sp=10
            , n_init_sample=5, isFlatSetting=False  )
_starttime=time.time()
opt.start_time=time.time()
xopt, fopt, _, eval_count = opt.run()
print(fopt,xopt, fopt, _, eval_count)
print(time.time()-_starttime)
'''

trial=Trials()
fmin = fmin(new_obj,HPOspace, max_evals=21, algo=tpe.suggest, trials=trial)
print(fmin)'''