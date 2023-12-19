import os
import time

import pandas as pd
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
    csv_list = os.listdir(data_path)
    for p in range(6, 7):
        # print(csv_list)
        for csv_file in csv_list:
            # number_metrics = 6
            out_result_RFR = []
            out_result_RR = []
            out_result_linear = []

            fold_path = path + "/" + csv_file
            print("Processing........ \n Filename: " + csv_file)

            csv_list1 = os.listdir(fold_path)
            train_path = fold_path + '/' + csv_list1[0]
            test_path = fold_path + '/' + csv_list1[1]

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)


            training_data_x, training_data_y = split_features_label(train_data)
            Cla_training_data_y = [1 if y > 0 else 0 for y in training_data_y]
            if flag == 'number':
                new_training_data_x, new_training_data_y = split_features_label_density(training_data_x,
                                                                                        training_data_y, 10)
            else:
                new_training_data_x, new_training_data_y = split_features_label_density(training_data_x,
                                                                                        Cla_training_data_y, 10)
            traincodeN = training_data_x[:, 10]
            cost = traincodeN
            print(f"--------begin training model: metric = {metrics}----------")
            start = time.time()

            de = LTR(X=new_training_data_x, y=new_training_data_y, cost=cost, costflag='loc',
                     logorlinear=ltrflag, metircs=metrics)
            Ltr_w = de.process()
            end = time.time()
            print("-------training time：", end - start, "-----------")

            if not os.path.exists(model_path):
                os.makedirs(model_path)
            modelsavepath = os.path.join(model_path,
                                         'EALTRModel_' + csv_file + '_version_' + str(p) + '.pkl')
            joblib.dump(Ltr_w, modelsavepath)



if __name__ == '__main__':
    flag1 = 'number'
    flag2 = 'without_number'
    # 读文件
    path = '../crossrelease_csv'
    model_path1 = '../Models/wn_crossrelease/LTR-linear/'

    model_path3 = '../Models/wn_crossrelease/DEJIT/'
    model_path4 = '../Models/crossrelease/DEJIT/'
    #without number
    #LTR-linear
    #ltr_train_save_model(path, model_path1, flag2, 'linear', "pofb20")

    #without number + DEJIT
    #ltr_train_save_model(path, model_path3, flag2, 'linear', 'dpa')
    # number + LTR-log
    model_path2 = '../Models/number_crossrelease/LTR-log/'
    #ltr_train_save_model(path, model_path2, flag1, 'log', "pofb20")

    # LTR-log
    #without number + LTR-log
    #ltr_train_save_model(path, model_path2, flag2, 'log', "pofb20")
    #without number + LTR-linear
    model_path5 = '../Models/wn_crossrelease/LTR-linear/'
    ltr_train_save_model(path, model_path5, flag2, 'linear', "pofb20")
    model_path6 = '../Models/wn_crossrelease/LTR-log/'
    ltr_train_save_model(path, model_path6, flag2, 'log', "pofb20")



