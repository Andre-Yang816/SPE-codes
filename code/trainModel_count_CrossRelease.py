import math
import os
import time
import warnings
import pandas as pd
import numpy as np
import joblib
from LTR_New import LTR
from PerformanceMeasure import PerformanceMeasure
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
warnings.filterwarnings('ignore')


header = ["RFR", "RR", "LTR-linear"]
#header = ["LTR-linear"]
RFR_tuned_parameters = [{'n_estimators': [10, 20, 30, 40, 50]}]
ridge_tuned_parameters = [{'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001]}]
cv_times = 3


if __name__ == '__main__':

    flags = ['number', 'without_number']
    path = '../crossrelease_csv'
    csv_list = os.listdir(path)


    print("begin time：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = time.time()
    for flag in flags:
        LR_list = []
        RR_list = []
        LTR_linear_list = []
        model_n = 0
        for file in csv_list:
            # number_metrics = 6
            out_result_RFR = []
            out_result_RR = []
            out_result_linear = []

            fold_path = path + "/" + file
            print("Processing........ \n Filename: " + file)
            # ant1.3_ant1.4 可用作输出文件的结果
            csv_list1 = os.listdir(fold_path)
            train_path = fold_path + '/' + csv_list1[0]
            test_path = fold_path + '/' + csv_list1[1]

            train_data = np.array(pd.read_csv(train_path))
            trainX = train_data[:, :-1]
            trainY = train_data[:, -1].astype(int)
            test_data = np.array(pd.read_csv(test_path))
            testX = test_data[:, :-1]
            testY = test_data[:, -1].astype(int)

            loc_index = 10
            index_loc0 = np.where(trainX[:, loc_index] == 0)[0]
            trainX = np.delete(trainX, index_loc0, axis=0)
            trainY = np.delete(trainY, index_loc0, axis=0)
            print("train index:{0}".format(index_loc0))
            index_loc1 = np.where(trainX[:, loc_index] == 0)[0]
            testX = np.delete(testX, index_loc1, axis=0)
            testY = np.delete(testY, index_loc1, axis=0)

            print("test index:{0}".format(index_loc1))
            train_loc = trainX[:, loc_index]
            test_loc = testX[:, loc_index]
            trainX = np.delete(trainX, [loc_index], axis=1)
            testX = np.delete(testX, [loc_index], axis=1)

            Cla_training_data_y = [1 if y > 0 else 0 for y in trainY]

            for i in range(10):
                print(f"--------trainging model begin:----------")
                start = time.time()
                print('times: {0}....'.format(i))

                if flag == 'without_number':
                    training_y = Cla_training_data_y
                else:
                    training_y = trainY

                print("RFR")


                def my_fpa_score(realbug, predbug):
                    return PerformanceMeasure(realbug, predbug).FPA()


                my_score = my_fpa_score

                regr = GridSearchCV(RandomForestRegressor(), RFR_tuned_parameters, cv=cv_times,
                                    scoring=make_scorer(my_score, greater_is_better=True))
                # regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators = 100)
                regr.fit(trainX, training_y)

                RFR_model_path = '../Models_count/CrossRelease/' + flag + '/RFR/'
                if not os.path.exists(RFR_model_path):
                    os.makedirs(RFR_model_path)
                RFR_modelsavepath = os.path.join(RFR_model_path, 'RFR' + file + '_version_' + str(i) + '.pkl')
                joblib.dump(regr, RFR_modelsavepath)

                print("Ridge Regression")
                ridge = GridSearchCV(Ridge(), ridge_tuned_parameters, cv=cv_times,
                                     scoring=make_scorer(my_score, greater_is_better=True))
                ridge.fit(trainX, training_y)
                RR_model_path = '../Models_count/CrossRelease/' + flag + '/RR/'
                if not os.path.exists(RR_model_path):
                    os.makedirs(RR_model_path)
                RR_modelsavepath = os.path.join(RR_model_path, 'RR' + file + '_version_' + str(i) + '.pkl')
                joblib.dump(ridge, RR_modelsavepath)

                print("LTR-linear")
                cost = [1 for k in range(len(training_y))]
                de = LTR(X=trainX, y=training_y, cost=cost, costflag='module', logorlinear='linear')
                Ltr_linear = de.process()
                LTR_model_path = '../Models_count/CrossRelease/' + flag + '/LTR-linear/'
                if not os.path.exists(LTR_model_path):
                    os.makedirs(LTR_model_path)
                LTR_modelsavepath = os.path.join(LTR_model_path, 'LTR-linear' + file + '_version_' + str(i) + '.pkl')
                joblib.dump(Ltr_linear, LTR_modelsavepath)

                end = time.time()
                print("-------train time：", end - start, "-----------")