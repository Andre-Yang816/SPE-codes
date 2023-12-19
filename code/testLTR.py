import os
import re
import time
import pandas as pd

import numpy as np
import joblib

from LTR import LTR
from PerformanceMeasure1 import PerformanceMeasure


def split_features_label(original_data):
    original_data = original_data.iloc[:, :]

    original_data = np.array(original_data)

    k = len(original_data[0])

    original_data = np.array(original_data)
    original_data_X = original_data[:, 0:k - 1]

    original_data_y = original_data[:, k - 1]

    return original_data_X, original_data_y


def split_features_label_density(training_data_X, training_data_y, index):
    want_row = [i for i in range(len(training_data_X))]
    new_train_data_x = training_data_X[want_row]

    want_col = [j for j in range(0, index)] + [j for j in range(index + 1, len(training_data_X[0]))]
    new_train_data_x = new_train_data_x[:, want_col]

    loc_train = training_data_X[:, [index]].squeeze()

    new_train_data_y = [i for i in training_data_y]

    for i in range(len(training_data_y)):
        if loc_train[i] > 0.0:
            new_train_data_y[i] = training_data_y[i] / loc_train[i]
        else:
            new_train_data_y[i] = 0

    return new_train_data_x, new_train_data_y


def locPerformance(real_list=None, pred_list=None, loc=None, lineofLOC=3000, ranking="defect", cost="module"):
    real = real_list
    pred = pred_list
    loc = loc
    ranking = ranking
    cost = cost
    if (len(pred) != len(real)) or (len(pred) != len(loc) or (len(loc) != len(real))):
        print("The predicted number or density of defects is inconsistent with the actual number or "
              "density of defects, and the input length is inconsistent!")
        exit()

    if (len(pred) != len(real)) or (len(pred) != len(loc) or (len(loc) != len(real))):
        exit()

    M = len(real)
    L = sum(loc)
    P = sum([1 if i > 0 else 0 for i in real])
    m = None
    Q = sum(real)

    if (ranking == "density" and cost == 'loc'):

        density = []
        for i in range(len(pred)):
            if loc[i] != 0:
                density.append(pred[i] / loc[i])
            else:
                density.append(-300000000)

        sort_axis = np.argsort(density)[::-1]

        sorted_pred = np.array(pred)[sort_axis]
        sorted_real = np.array(real)[sort_axis]
        sorted_loc = np.array(loc)[sort_axis]
        print(sorted_loc[:10])
        locOfPercentage = lineofLOC
        print("locOfPercentage=", 0.2*L)
        # print("sorted_1=", sorted_loc[0])
        sum_ = 0
        for i in range(len(sorted_loc)):
            sum_ += sorted_loc[i]
            if (sum_ > locOfPercentage):
                m = i
                break
            elif (sum_ == locOfPercentage):
                m = i + 1
                break
        #print('module number:'+str(m))
        PMI = m / M

    # print("PMI=", PMI)
    tp = sum([1 if sorted_real[j] > 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
    fn = sum([1 if sorted_real[j] > 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
    fp = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
    tn = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
    # print('tp:{0},fn:{1},fp:{2},tn:{3}'.format(tp,fn,fp,tn))
    if (tp + fn + fp + tn == 0):
        Precisionx = 0
    else:
        Precisionx = (tp + fn) / (tp + fn + fp + tn)

    if (P == 0):
        Recallx = 0
    else:
        Recallx = (tp + fn) / P

    if (P == 0 or PMI == 0):
        recallPmi = 0
    else:
        recallPmi = Recallx / PMI

    if (Recallx + Precisionx == 0):
        F1x = 0
    else:
        F1x = 2 * Recallx * Precisionx / (Recallx + Precisionx)

    IFLA = 0
    IFMA = 0

    for i in range(m):
        if (sorted_real[i] > 0):
            break
        else:
            IFLA += sorted_loc[i]

            IFMA += 1

    PofB = sum([sorted_real[j] if sorted_real[j] > 0 else 0 for j in range(m)]) / Q

    if (Q == 0 or PMI == 0):
        PofBPmi = 0
    else:
        PofBPmi = PofB / PMI

    return F1x, PMI, PofB, PofBPmi

def writeToFile(name, list, flag):
    header_name = ['Precision@20%', 'Recall@20%', 'F1@20%', 'PMI@20%', 'PofB@20%', 'PofB/PMI@20%', 'Popt', 'IFA',
                   'F1@3000', 'PMI@3000', 'PofB@3000', 'PofB/PMI@3000',  'F1@5000', 'PMI@5000', 'PofB@5000', 'PofB/PMI@5000']

    dir = '../output_Density/crossProject/{0}/'.format(flag)
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = dir + name + '.csv'
    df = pd.DataFrame(list, columns=header_name)
    df.to_csv(path, index=False)

def ltr_prediction(data_path, model_path, Q=[0], P=[1 / 8], percent=0.2):
    resultlist = []
    model_count = 0
    for q in Q:
        for p in P:
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

                    training_data_x, training_data_y = split_features_label(
                        dataset_train)
                    testing_data_x, testing_data_y = split_features_label(
                        dataset_test)
                    new_training_data_x, new_training_data_y = split_features_label_density(
                        training_data_x,
                        training_data_y, 10)
                    new_testing_data_x, new_testing_data_y = split_features_label_density(
                        testing_data_x,
                        testing_data_y, 10)

                    traincodeN = training_data_x[:, 10]

                    train_LOC = []
                    for i in range(len(training_data_y)):
                        if training_data_y[i] == 0:
                            train_LOC.append(traincodeN[i])
                    train_LOC_sort_index = np.argsort(train_LOC)

                    cost = traincodeN

                    # modelsavepath = os.path.join(model_path,
                    #                              'RankingSVMModel_' + file + '.pkl')
                    # print('model name:'+modelsavepath)
                    # model_svm = joblib.load(modelsavepath)
                    # Ltr_linear_pred = model_svm.predict(new_testing_data_x)
                    modelsavepath = os.path.join(model_path, 'EALTRModel_version_1.pkl')
                    #
                    modelsavepath = os.path.join(model_path,
                                                 'EALTRModel_' + file + '_version_1.pkl')
                    Ltr_w = joblib.load(modelsavepath)

                    de = LTR(X=new_training_data_x, y=new_training_data_y, cost=cost, costflag='loc',
                              logorlinear='linear')
                    testingcodeN = testing_data_x[:, 10]
                    Ltr_linear_pred = de.predict(new_testing_data_x, Ltr_w)
                    EALTR_pred = np.array(Ltr_linear_pred)
                    EALTR_pred_topIndex = np.argsort(-EALTR_pred)

                    compared_loc = 0
                    index = len(train_LOC) * p
                    yu = len(train_LOC) % (1 / p)
                    if yu != 0:
                        compared_loc = train_LOC[train_LOC_sort_index[int(index)]]
                    else:
                        compared_loc = (train_LOC[train_LOC_sort_index[int(index)]] + train_LOC[
                            train_LOC_sort_index[int(index) - 1]]) / 2

                    print("###compared_loc###")
                    print(compared_loc)

                    EALTR_trainingdata_pred = de.predict(new_training_data_x, Ltr_w)
                    EALTR_trainingdata = []
                    for i in range(len(EALTR_trainingdata_pred)):
                        EALTR_trainingdata.append(EALTR_trainingdata_pred[i] * traincodeN[i])

                    # measure
                    traindataPrecisionx, traindataRecallx, traindataF1x, traindataIFMA, traindataPMI, traindatarecallPmi, traindataPofB, traindataPofbpmi = PerformanceMeasure(
                        training_data_y, EALTR_trainingdata, traincodeN, 0.2, 'density',
                        'loc').Performance()
                    print("traindataPrecisionx = " + str(traindataPrecisionx))
                    print("traindataIFMA = " + str(traindataIFMA))

                    if traindataPrecisionx < 0.1 or traindataIFMA > 1:
                        print("=====需要进行重排名=====")
                        for j in range(q):
                            topIndexNum = EALTR_pred_topIndex[j]
                            if testingcodeN[topIndexNum] < compared_loc:
                                topIndexValue = Ltr_linear_pred[topIndexNum]
                                Ltr_linear_pred[topIndexNum] = topIndexValue - 300000000

                    LTR_DP_lTR = []
                    for j in range(len(Ltr_linear_pred)):
                        LTR_DP_lTR.append(Ltr_linear_pred[j] * testingcodeN[j])

                    Precisionx, Recallx, F1x, IFMA, PMI, recallPmi, PofB, Pofbpmi = PerformanceMeasure(
                        testing_data_y, LTR_DP_lTR,
                        testingcodeN, percent,
                        'density',
                        'loc').Performance()
                    wholenormOPT = PerformanceMeasure(testing_data_y, LTR_DP_lTR, testingcodeN, percent,
                                                      'density',
                                                      'loc').POPT()

                    F1x3000module, PMI3000module, PofB3000module, PofBPMI3000module = locPerformance(testing_data_y, LTR_DP_lTR, testingcodeN, 3000,
                                                      'density',
                                                      'loc')
                    F1x5000module, PMI5000module, PofB5000module, PofBPMI5000module = locPerformance(testing_data_y, LTR_DP_lTR, testingcodeN, 5000,
                                                      'density',
                                                      'loc')
                    # dataset = trainingfile + testingfile
                    out_result_svm = [Precisionx, Recallx, F1x, PMI, PofB, Pofbpmi, wholenormOPT, IFMA,
                                      F1x3000module, PMI3000module, PofB3000module, PofBPMI3000module,
                                      F1x5000module, PMI5000module, PofB5000module, PofBPMI5000module]

                    resultlist.append(out_result_svm)

                    print("**********Recallx**********")
                    print(Recallx)
                    print("**********Precisionx**********")
                    print(Precisionx)
                    print("**********PMI**********")
                    print(PMI)
                    print("**********IFMA**********")
                    print(IFMA)
                    print("**********F1x**********")
                    print(F1x)
                    print("**********wholenormOPT**********")
                    print(wholenormOPT)
                    print("**********PofB**********")
                    print(PofB)
                    print("**********PofBpmi**********")
                    print(Pofbpmi)
                    print('metric@3000')
                    print(F1x3000module, PMI3000module, PofB3000module, PofBPMI3000module)
                    print('metric@5000')
                    print(F1x5000module, PMI5000module, PofB5000module, PofBPMI5000module)
    writeToFile('EALTR_', resultlist, '1')


if __name__ == '__main__':
    data_path = '../CrossProjectData'
    model_path = '../Models/LTR/'
    ltr_prediction(data_path, model_path)