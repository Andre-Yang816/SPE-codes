import math
import os
import time
import warnings
import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
from LTR_New import LTR
from PerformanceMeasure import PerformanceMeasure
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
warnings.filterwarnings('ignore')


header = ["LR", "RR", "LTR-linear"]
ridge_tuned_parameters = [{'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001]}]
lr_tuned_parameters = [{'normalize': [True, False]}]
cv_times = 3


if __name__ == '__main__':

    flags = ['number', 'without_number']
    #flags = ['without_number']
    path = '../CrossProjectData'
    csv_list = os.listdir(path)


    print("begin time：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = time.time()
    for flag in flags:
        LR_list = []
        RR_list = []
        LTR_linear_list = []
        for root, dirs, files, in os.walk(path):
            for file in files:
                if file != 'xerces-1.1.csv':
                    continue
                print("Processing...\nFile name:" + file)
                out_result_LR = []
                out_result_RR = []
                out_result_linear = []
                # Get test set paths and data
                file_path = os.path.join(path, file)
                dataset_test = pd.read_csv(file_path)
                # Obtain training set paths and data
                dataset_train = pd.DataFrame(columns=dataset_test.columns)
                for tmp_file in files:
                    if tmp_file == file:
                        continue
                    else:
                        tmp_file_path = os.path.join(path, tmp_file)
                        tmp_df = pd.read_csv(tmp_file_path)
                        dataset_train = pd.concat([dataset_train, tmp_df])

                train_data = np.array(dataset_train)
                trainX = train_data[:, :-1].astype(float)
                trainY = train_data[:, -1].astype(int)
                test_data = np.array(dataset_test)
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
                train_loc = trainX[:, loc_index].astype(int)
                test_loc = testX[:, loc_index].astype(int)
                trainX = np.delete(trainX, [loc_index], axis=1)
                testX = np.delete(testX, [loc_index], axis=1)

                Cla_training_data_y = [1 if y > 0 else 0 for y in trainY]

                for i in range(1):
                    print(f"--------开始训练模型:----------")
                    start = time.time()
                    print('times: {0}....'.format(i))

                    if flag == 'without_number':
                        training_y = Cla_training_data_y
                    else:
                        training_y = trainY

                    print("Linear Regression")


                    def my_fpa_score(realbug, predbug):
                        return PerformanceMeasure(realbug, predbug).FPA()


                    my_score = my_fpa_score
                    lr = GridSearchCV(linear_model.LinearRegression(), lr_tuned_parameters, cv=cv_times,
                                      scoring=make_scorer(my_score, greater_is_better=True))
                    lr.fit(trainX, training_y)
                    LR_model_path = '../Models_count/CrossProject/' + flag + '/LR/'
                    if not os.path.exists(LR_model_path):
                        os.makedirs(LR_model_path)
                    LR_modelsavepath = os.path.join(LR_model_path, 'LR' + file + '_version_' + str(i) + '.pkl')
                    joblib.dump(lr, LR_modelsavepath)

                    print("Ridge Regression")
                    ridge = GridSearchCV(Ridge(), ridge_tuned_parameters, cv=cv_times,
                                         scoring=make_scorer(my_score, greater_is_better=True))
                    ridge.fit(trainX, training_y)
                    RR_model_path = '../Models_count/CrossProject/' + flag + '/RR/'
                    if not os.path.exists(RR_model_path):
                        os.makedirs(RR_model_path)
                    RR_modelsavepath = os.path.join(RR_model_path, 'RR' + file + '_version_' + str(i) + '.pkl')
                    joblib.dump(ridge, RR_modelsavepath)

                    print("LTR-linear")
                    cost = [1 for k in range(len(training_y))]
                    de = LTR(X=trainX, y=training_y, cost=cost, costflag='module', logorlinear='linear')
                    Ltr_linear = de.process()
                    LTR_model_path = '../Models_count/CrossProject/' + flag + '/LTR-linear/'
                    if not os.path.exists(LTR_model_path):
                        os.makedirs(LTR_model_path)
                    LTR_modelsavepath = os.path.join(LTR_model_path,
                                                     'LTR-linear' + file + '_version_' + str(i) + '.pkl')
                    joblib.dump(Ltr_linear, LTR_modelsavepath)

                    end = time.time()
                    print("-------训练时间：", end - start, "-----------")

