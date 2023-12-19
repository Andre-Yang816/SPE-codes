import os
import time
import pandas as pd
import numpy as np
import joblib
from LTR import LTR
from PerformanceMeasure1 import PerformanceMeasure
from testLTR import split_features_label_density, locPerformance


def testLTR(data_path, model_path, flag, model, target, savepath, modelname):
    output_result = []
    for root, dirs, files, in os.walk(data_path):
        for file in files:

            print("Processing...\nFile name:" + file)
            out_result_svm = []

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

            # nn-filter
            #trainX, trainY = NN_filter(trainX, trainY, testX)

            # 取出loc列
            loc_index = 10
            # 先把loc==0的行删除
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
            #Cla_testing_data_y = [1 if y > 0 else 0 for y in testY]
            testing_data_y = testY.tolist()
            temp_result = [0] * 16
            # if model != 'log':
            #     rangeNumber = [i for i in range(1,11)]
            # else:
            #     rangeNumber = [i for i in range(10)]
            if flag == 'number':
                new_training_data_x, new_training_data_y = split_features_label_density(trainX,
                                                                                        trainY, 10)

            else:
                new_training_data_x, new_training_data_y = split_features_label_density(trainX,
                                                                                        Cla_training_data_y, 10)
            for r in range(10):
                if flag == 'number' and modelname in ['LTR-linear', 'DEJIT']:
                    #control model name
                    if r == 0:
                        r = 10
                    new_file = file[:-4] + '-training'
                    modelsavepath = os.path.join(model_path,
                                                 'EALTRModel_' + new_file + '.csv_version_' + str(r) + '.pkl')
                else:
                    modelsavepath = os.path.join(model_path,
                                                 'EALTRModel_' + file + '_version_' + str(r) + '.pkl')


                Ltr_w = joblib.load(modelsavepath)

                cost = train_loc
                de = LTR(X=trainX, y=new_training_data_y, cost=cost, costflag='loc', logorlinear=model, metircs=target)

                Ltr_linear_pred = de.predict(testX, Ltr_w)

                rankingLTR_predN = []
                for j in range(len(Ltr_linear_pred)):
                    rankingLTR_predN.append(Ltr_linear_pred[j] * test_loc[j])

                Precisionx, Recallx, F1x, IFMA, PMI, recallPmi, PofB, Pofbpmi = PerformanceMeasure(
                    testing_data_y, rankingLTR_predN,
                    test_loc, 0.2,
                    'density',
                    'loc').Performance()

                wholenormOPT = PerformanceMeasure(testing_data_y, rankingLTR_predN, test_loc, 0.2, 'density', 'loc').POPT()
                F1x3000module, PMI3000module, PofB3000module, PofBPMI3000module = locPerformance(testing_data_y,
                                                                                                 rankingLTR_predN,
                                                                                                 test_loc, 3000,
                                                                                                 'density', 'loc')
                F1x5000module, PMI5000module, PofB5000module, PofBPMI5000module = locPerformance(testing_data_y,
                                                                                                 rankingLTR_predN,
                                                                                                 test_loc, 5000,
                                                                                                 'density', 'loc')
                temp_result1 = [Precisionx, Recallx, F1x, PofB, PMI, Pofbpmi, wholenormOPT, IFMA,
                                F1x3000module, PofB3000module, PMI3000module, PofBPMI3000module,
                                F1x5000module, PofB5000module, PMI5000module, PofBPMI5000module]
                temp_result = np.sum([temp_result, temp_result1], axis=0).tolist()
            temp_result = [item / 10 for item in temp_result]
            output_result.append(temp_result)
            

    writeToFile(modelname, output_result, savepath)

def writeToFile(name, list, output_path):
    header_name = ['Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PMI@20%',  'PofB/PMI@20%', 'Popt', 'IFA',
                   'F1@3000', 'PofB@3000', 'PMI@3000', 'PofB/PMI@3000',
                   'F1@5000', 'PofB@5000', 'PMI@5000', 'PofB/PMI@5000']

    dir = output_path
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = dir + name + '.csv'
    df = pd.DataFrame(list, columns=header_name)
    df.to_csv(path, index=False)

if __name__ == '__main__':

    # 读文件
    flags = ['number', 'without_number']
    path = '../CrossProjectData'
    model_path1 = '../Models/number_crossproject/LTR-linear/'
    model_path2 = '../Models/number_crossproject/LTR-log/'
    model_path3 = '../Models/number_crossproject/DEJIT/'
    model_path4 = '../Models/wn_crossproject/LTR-linear/'
    model_path5 = '../Models/wn_crossproject/LTR-log/'
    model_path6 = '../Models/wn_crossproject/DEJIT/'

    models = ['LTR-linear', 'LTR-log', 'DEJIT']

    for flag in flags:
        for model in models:
            if model == 'LTR-linear':
                if flag == 'number':
                    model_path = model_path1
                else:
                    model_path = model_path4
                logorlinear = 'linear'
                target = 'pofb20'
            elif model == 'LTR-log':
                if flag == 'number':
                    model_path = model_path2
                else:
                    model_path = model_path5
                logorlinear = 'log'
                target = 'pofb20'
            else:
                if flag == 'number':
                    model_path = model_path3
                else:
                    model_path = model_path6
                logorlinear = 'linear'
                target = 'dpa'

            # LTR-linear linear+pofb
            # testSVM(path, model_path1, flag1, model1, target1)
            # LTR-log
            # testSVM(path, model_path2, flag1, model2, target1)
            # DEJIT
            # testSVM(path, model_path3, flag1, model1, target2)

            save_path = '../output_Density_final/crossProject/'+flag+'/'
            print('Save to ...'+save_path)
            testLTR(path, model_path, flag, logorlinear, target, save_path, model)