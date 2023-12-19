import os
import time

import pandas as pd
import numpy as np
import joblib
from LTR import LTR
from testLTR import split_features_label, split_features_label_density


def writeToFile(name, list, flag):
    header_name = ['Precision@20%', 'Recall@20%', 'F1@20%', 'PMI@20%', 'PofB@20%', 'PofB/PMI@20%', 'Popt', 'IFA',
                   'F1@1000', 'PMI@1000', 'PofB@1000', 'PofB/PMI@1000',  'F1@2000', 'PMI@2000', 'PofB@2000', 'PofB/PMI@2000']

    dir = '../output_Density/crossProject/{0}/'.format(flag)
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = dir + name + '.csv'
    df = pd.DataFrame(list, columns=header_name)
    df.to_csv(path, index=False)


def ltr_train_save_model(data_path, model_path, flag, ltrflag='linear', metrics="pofb20"):
    for p in range(10):
        for root, dirs, files, in os.walk(data_path):
            for file in files:
                print("Processing...\nFile name:" + file)
                out_result_svm = []

                # Get test set paths and data
                file_path = os.path.join(data_path, file)
                dataset_test = pd.read_csv(file_path)
                # Obtain training set paths and data
                dataset_train = pd.DataFrame(columns=dataset_test.columns)
                for tmp_file in files:
                    if tmp_file == file:
                        continue
                    else:
                        tmp_file_path = os.path.join(data_path, tmp_file)
                        tmp_df = pd.read_csv(tmp_file_path)
                        dataset_train = pd.concat([dataset_train, tmp_df])
                training_data_x, training_data_y = split_features_label(dataset_train)
                Cla_training_data_y = [1 if y > 0 else 0 for y in training_data_y]
                if flag == 'number':
                    new_training_data_x, new_training_data_y = split_features_label_density(training_data_x,training_data_y, 10)
                else:
                    new_training_data_x, new_training_data_y = split_features_label_density(training_data_x,
                                                                                            Cla_training_data_y, 10)
                traincodeN = training_data_x[:, 10]
                cost = traincodeN
                print(f"--------training begin: metric = {metrics}----------")
                start = time.time()
                de = LTR(X=new_training_data_x, y=new_training_data_y, cost=cost, costflag='loc',
                         logorlinear=ltrflag, metircs=metrics)
                Ltr_w = de.process()
                end = time.time()
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                print("-------training time：", end - start, "-----------")
                modelsavepath = os.path.join(model_path,
                                             'EALTRModel_' + file + '_version_' + str(p) + '.pkl')
                joblib.dump(Ltr_w, modelsavepath)



if __name__ == '__main__':
    flag1 = 'number'
    flag2 = 'without_number'

    path = '../CrossProjectData'
    model_path1 = '../Models/LTR-linear/'
    model_path2 = '../Models/LTR-log/'
    model_path3 = '../Models/DEJIT/'

    ltr_train_save_model(path, model_path1, flag2, 'linear', "pofb20")
    #ltr_train_save_model(path, model_path2, flag2, 'log', "pofb20")
    #ltr_train_save_model(path, model_path3, flag2, 'linear', 'dpa')

    #model_path4 = '../Models/number_crossproject/LTR-log/'
    #ltr_train_save_model(path, model_path4, flag1, 'log', "pofb20")

    #testModels(path,'../Models/number_crossproject/LTR-linear/',flag1)

