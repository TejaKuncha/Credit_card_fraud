from time import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics.regression import r2_score
from sklearn.metrics import f1_score


t0 = time()
warnings.filterwarnings('ignore')

print('Load Data')
data = pd.read_csv('creditcard.csv', sep=',', header=0)
colofans = ['Amount', 'Class']
labels = data[colofans]
data = data.drop(colofans, axis=1)

print('Split Data')
Xtrain, Xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.25, random_state=77)
train_class = ytrain['Class']
test_class = ytest['Class']
train_reg = ytrain['Amount']
test_reg = ytest['Amount']

print('Process data for KNN Regressor')
scale = MinMaxScaler()
pca = PCA(n_components=20)
Xtrain = scale.fit_transform(Xtrain)
Xtest = scale.transform(Xtest)
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)
train_reg = scale.fit_transform(np.reshape(train_reg, (-1, 1)))
test_reg = scale.transform(np.reshape(test_reg, (-1, 1)))

knr = KNeighborsRegressor(n_neighbors=3)
knc = KNeighborsClassifier(n_neighbors=9)
svr = SGDRegressor()
svm = SVC(kernel='poly', degree=2, coef0=1, cache_size=5000, decision_function_shape='ovr')
mlr = MLPRegressor(hidden_layer_sizes=(32, 8), activation='relu', solver='adam')
mlc = MLPClassifier(hidden_layer_sizes=(32, 8), activation='relu', solver='adam')
rtc = RandomForestClassifier(n_estimators=10)
rtr = RandomForestRegressor(n_estimators=10)
regressors = {'Kneighbor_regressor': knr, 'SV_regressor': svr, 'MLP_regressor': mlr, 'RTree_regressor': rtr}
classifiers = {'Kneighbor_classifier': knc, 'SV_classifier': svm, 'MLP_classifier': mlc, 'RTree_classifier': rtc}
estimator_class = SGDClassifier()
estimator_reg = SGDRegressor()
refer = RFE(estimator=estimator_reg, n_features_to_select=5)
refec = RFECV(estimator_class, cv=5)

refer.fit(Xtrain, train_reg)
Xtrain1 = refer.transform(Xtrain)
Xtest1 = refer.transform(Xtest)
print('RFE score = ', refer.score(Xtrain, train_reg))
print('No. of features = ', refer.n_features_)
X_train = np.concatenate((Xtrain1, train_reg), axis=1)
reg_scores = []

for name in regressors:
    print('\n', name)
    class_scores = []
    reg = regressors[name]
    reg.fit(Xtrain1, train_reg)
    y = reg.predict(Xtest1)
    X_test = np.concatenate((Xtest1, y.reshape((-1, 1))), axis=1)
    print('Reg_score = ', r2_score(test_reg, y))
    for clame in classifiers:
        print(clame)
        clf = classifiers[clame]
        clf.fit(X_train, train_class)
        y_ = clf.predict(X_test)
        class_scores.append(f1_score(test_class, y_))
    reg_scores.append(class_scores)
print(*[name for name in regressors])
for i in reg_scores:
    print(*i)

print('\n Time take for the code to run ==> %.2f sec' % (time()-t0))
