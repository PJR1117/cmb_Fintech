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
from scipy.stats import boxcox
from sklearn.impute import KNNImputer
from sklearn.metrics import recall_score, confusion_matrix

class my_tool():
    def __init__(self,estimator):
        self.estimator = estimator
        self.cv = 5


    #EDA
    def eda(self,data):
        # 是否平衡
        print("是否平衡")
        print(pd.value_counts(data["LABEL"]))
        print('-' * 50)
        # 字段类型
        print("字段类型")
        print(data.info())
        print('-' * 50)
        # 分布情况
        print("分布情况")
        print(data.describe())
        print('-' * 50)

    # 学习曲线
    def plot_learning_curve(self, title, X, y, ylim=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):  # n_jobs表示是否进行多进程处理
        estimator = self.estimator
        cv = self.cv
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

    # 绘制混淆矩阵图
    def plot_confusion_matrix(cm, classes,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        :param cm: 由测试集标签和测试集预测结果组成的混淆矩阵
        :param classes: 标签的格式 如0,1
        :param title: 标题
        :param cmap:
        :return:
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


    # 验证曲线
    def plot_validation_curve(self, X, y, param_name, param_range, ylim=None, ):
        estimator = self.estimator
        cv = self.cv
        train_scores, test_scores = validation_curve(estimator=estimator, X=X, y=y, param_name=param_name,
                                                     param_range=param_range, cv=cv)
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

    # ROC曲线:
    def plot_roc_curve(self,X_test, y_test):
        estimator = self.estimator
        fpr, tpr, threshold = roc_curve(y_test, estimator.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
        print(round(auc, 5))

#数据清洗
    #删除缺失值
    def drop_na(self,data, prop, thresh):
        if isinstance(data, pd.DataFrame):
            for i in data.columns:
                prob_null = sum(pd.isnull(data[i])) / len(data.index)
                if prob_null >= prop:
                    del data[i]
        data = data.dropna(thresh=thresh)
        return data

    #众数填充
    def fill_nan_with_mode(self,data):
        col_index = data.columns
        mode_list = []
        for i in col_index:
            mode = data[i].mode()[0]
            mode_list.append(mode)
        dict_results = dict(zip(col_index, mode_list))
        data.fillna(value=dict_results, inplace=True)
        return data
    #KNN填充


    #随机森林填充



    #正态标准化
    def Normalization(self,data):
        columns = data.dtypes[data.dtypes == "float64"].index
        data_sub = data.loc[:,data.dtypes[data.dtypes != "float64"].index]
        data = data[columns]
        data = data.aggregate(lambda x:(x-x.mean())/x.std())
        data = pd.merge(data,data_sub,right_index=True,left_index=True)
        return data

    #将分类变量转化为哑变量
    def get_categorial_data(self,data):
        fill_nan_with_mode = self.fill_nan_with_mode()
        index = data.dtype[data.dtypes == 'object'].index
        data_obj = data.loc[:,index].aggregate(lambda x:fill_nan_with_mode(x))
        for i in data_obj.columns:
            data_obj.loc[:, i] = pd.Categorical(data_obj[i])
            del data[i]
        # data = data.aggregate(lambda x:minmax(x))
        data_obj = pd.get_dummies(data_obj, drop_first=True)
        data = pd.merge(data, data_obj, right_index=True, left_index=True)
        return data

    #minmax变换
    def minmax(self,data):
        scal = data.max() - data.min()
        data_transformed = (data - data.min()) / scal
        return data_transformed

    # 对非obj进行boxcox变化
    def get_boxcox(self,data):
        minmax = self.minmax()
        columns = data.dtypes[data.dtypes == "float64"].index
        data_sub = data.loc[:, data.dtypes[data.dtypes != "float64"].index]
        data = data[columns].aggregate(lambda x:minmax(x))
        data = data.aggregate(lambda x: boxcox(x + 1)[0])
        data = pd.merge(data, data_sub, left_index=True, right_index=True)
        return data

    #生成多项式
    def get_ploy(self,data,incelude_bias=True,degree=2):
        ploy = PolynomialFeatures(degree=degree, include_bias=incelude_bias)
        X_train_ploy = ploy.fit_transform(data)
        name_list = ploy.get_feature_names()
        X_train_ploy = pd.DataFrame(X_train_ploy,columns=name_list)
        return X_train_ploy

    #嵌入法选择(针对树模型)
    def select_embedded(self,X,y):
        estimator = self.estimator
        importance = estimator.fit(X, y).feature_importances_.max()
        treshold = np.linspace(0, importance, 20)
        score = []
        for i in treshold:
            selector = SelectFromModel(estimator, threshold=i).fit(X, y)
            X_embedded = selector.transform(X)
            once = cross_val_score(estimator, X_embedded, y, cv=5).mean()
            score.append(once)
        plt.plot(treshold, score)
        plt.show()

    # #互信息法
    # def select_mic(X_train, y_train, treshold=0):
    #     result = MIC(X_train, y_train)
    #     k = result.shape[0] - sum(result <= treshold)
    #     selector = SelectKBest(MIC, k).fit(X_train, y_train)
    #     X_fsmic = selector.transform(X_train, y_train)
    #     return X_fsmic
    #
    # #方差过滤法
    # def select_var(estimator, X, y, X_test):
    #     selector = VarianceThreshold(threshold=np.median(X.var())).fit(X)
    #     X_train_var = selector.transform(X)
    #     X_test_var = selector.transform(X_test)
    #     score = cross_val_score(estimator, X, y=y, cv=5).mean()
    #     return X_train_var, X_test_var, score

    #pca
    def select_pca(self,X,y):
        estimator = self.estimator
        score = []
        for i in range(X.shape[1]):
            pca = PCA(n_components=i).fit(X)
            new_X = pca.transform(X)
            once = cross_val_score(estimator,new_X,y,cv=5).mean()
            score.append(once)
        plt.plot(range(X.shape[1]),score)
        plt.show()


