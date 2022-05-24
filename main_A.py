import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from  sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve,auc,roc_auc_score
from xgboost import XGBClassifier as xgb
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE

#todo 准备性能测量曲线:

#学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)): #n_jobs表示是否进行多进程处理
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    plt.show()
    return plt
#验证曲线
def plot_validation_curve(estimator, X, y, param_name, param_range, ylim=None, cv=None,):
    train_scores, test_scores = validation_curve(estimator=estimator, X=X, y=y, param_name=param_name, param_range=param_range, cv=cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel('Parameter')
    plt.ylabel('Score')
    plt.ylim(ylim)
    plt.show()
#ROC曲线:
def plot_roc_curve(X_test,y_test,estimator):
    fpr,tpr,threshold = roc_curve(y_test,estimator.predict_proba(X_test)[:,1])
    auc = roc_auc_score(y_test,estimator.predict_proba(X_test)[:,1])
    plt.plot(fpr,tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    print(round(auc,5))
#todo 数据清洗:

# _data = pd.read_excel('data/train.xlsx',na_values='?')
# data = _data.iloc[:, 1:]
def drop_na(data,prop,thresh):
    if isinstance(data,pd.DataFrame):
        for i in data.columns:
            prob_null = sum(pd.isnull(data[i])) / len(data.index)
            if prob_null >= prop:
                del data[i]
    data = data.dropna(thresh=thresh)
    return data
# data = drop_na(data,0.25,30)
# data = data.dropna(how='any')
# col_index = data.columns
# data['NB_CTC_HLD_IDV_AIO_CARD_SITU'] = pd.Categorical(data['NB_CTC_HLD_IDV_AIO_CARD_SITU'])
# data['WTHR_OPN_ONL_ICO'] = pd.Categorical(data['WTHR_OPN_ONL_ICO'])
# data = pd.get_dummies(data,drop_first=True)

#todo 缺失值处理
#先用众数填充
def fill_nan_with_mode(dataframe):
    col_index = dataframe.columns
    mode_list = []
    for i in col_index:
        mode = dataframe[i].mode()[0]
        mode_list.append(mode)
    dict_results = dict(zip(col_index,mode_list))
    dataframe.fillna(value=dict_results,inplace=True)
    return dataframe


#todo 训练测试集分离
# X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,1:],data.iloc[:,0])

#todo 选择分类器
estimator = RandomForestClassifier(n_estimators=100,random_state=0,max_features='log2',max_depth=25)
# todo 特征工程
#
# ploy = PolynomialFeatures(degree=2).fit(X_train)
# X_train_ploy = ploy.transform(X_train)
# y_train_ploy = y_train
# X_test_ploy = ploy.transform(X_test)
# y_test_ploy = y_test

#box-cox归一化

#多项式

#todo 特征选择,使得特征更加紧凑:
#先用方差过滤法,通过验证曲线选得最优参数:
def select_var(estimator,X,y,X_test):
    selector = VarianceThreshold(threshold=np.median(X.var())).fit(X)
    X_train_var = selector.transform(X)
    X_test_var = selector.transform(X_test)
    score = cross_val_score(estimator,X,y=y,cv=5).mean()
    return X_train_var,X_test_var,score

#再用互信息法
def select_mic(X_train,y_train,treshold=0):
    result = MIC(X_train,y_train)
    k = result.shape[0] - sum(result<=treshold)
    selector = SelectKBest(MIC,k).fit(X_train,y_train)
    X_fsmic = selector.transform(X_train,y_train)
    return X_fsmic

#用embedded嵌入法
def select_embedded(estimator,X,y):
    importance = estimator.fit(X,y).feature_importances_.max()
    treshold = np.linspace(0,importance,20)
    score = []
    for i in treshold:
        selector = SelectFromModel(estimator,threshold=i).fit(X,y)
        X_embedded = selector.transform(X)
        once = cross_val_score(estimator,X_embedded,y,cv=5).mean()
        score.append(once)
    plt.plot(treshold,score)
    plt.show()

# X_embedded = SelectFromModel(estimator,threshold=0.017).fit_transform(X_train,y_train)
# y_embedded = y_train


#pca
def select_pca(estimator,X,y):
    score = []
    for i in range(X.shape[1]):
        pca = PCA(n_components=i).fit(X)
        new_X = pca.transform(X)
        once = cross_val_score(estimator,new_X,y,cv=5).mean()
        score.append(once)
    plt.plot(range(X.shape[1]),score)
    plt.show()

# todo 网格搜索,确定参数:
# param_grid = {
#     'n_estimators':np.arange(1,201,10),
#     'max_features':['log2'],
#     'max_depth':np.arange(3,13),
#     'min_samples_leaf':np.arange(1,30),
#     'bootstrap':[True,False]
# }
# grid = GridSearchCV(estimator,param_grid=param_grid,cv=5).fit(X_train,y_train)
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_)
# print(grid.best_index_)

# #todo 输出概率:
# plot_learning_curve(estimator,title='learning curve',X=X_train,y=y_train,cv=5)
# plot_validation_curve(estimator,X_train,y_train,param_name='min_samples_leaf',param_range=[1,2,3,4,5,6,7,8,9,10],cv=5)

if __name__ == '__main__':
    # _test_data = pd.read_excel('./data/test_A榜.xlsx',na_values='?')
    # test_data = _test_data.loc[:,col_index[1:]]
    # test_data = fill_nan_with_mode(test_data)
    # test_data['NB_CTC_HLD_IDV_AIO_CARD_SITU'] = pd.Categorical(test_data['NB_CTC_HLD_IDV_AIO_CARD_SITU'])
    # test_data['WTHR_OPN_ONL_ICO'] = pd.Categorical(test_data['WTHR_OPN_ONL_ICO'])
    # test_data = pd.get_dummies(test_data,drop_first=True)
    # test_data = SelectFromModel(estimator, threshold=0.017).fit(X_train, y_train).transform(test_data)
    # # print(test_data.shape)
    # print(X_embedded.shape)
    # probility = estimator.fit(X_embedded,y_embedded).predict_proba(test_data)[:,1]
    # result = _test_data[['CUST_UID']]
    # result['probility'] = probility
    # print(result)
    # result.to_csv('results.txt',sep='\t',index=False,header=None)
    # plot_roc_curve(X_embedded,y_embedded,estimator)
    data = pd.read_excel('./data/train.xlsx',na_values='?')
    y = data[['LABEL']]
    X = data.drop(['CUST_UID','LABEL'],axis=1)
    smote = SMOTE(random_state=0)
    X_oversampled,y_oversampled = smote.fit_resample(X,y)
    print(pd.value_counts(y_oversampled))


