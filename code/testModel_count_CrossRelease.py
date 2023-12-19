import os
import time
import pandas as pd
import numpy as np
import joblib
from LTR import LTR
from PerformanceMeasure import PerformanceMeasure


def locPerformance(real_list=None, pred_list=None, loc=None, numberofLOC=30, ranking="defect", cost="module"):
    real = real_list
    pred = pred_list
    loc = loc
    ranking = ranking
    cost = cost
    if (len(pred) != len(real)) or (len(pred) != len(loc) or (len(loc) != len(real))):
        print("The predicted number or density of defects is inconsistent with the actual number or "
              "density of defects, and the input length is inconsistent!")
        exit()

    M = len(real)
    L = sum(loc)
    P = sum([1 if i > 0 else 0 for i in real])
    Q = sum(real)

    if (ranking == "defect" and cost == 'module'):
        sort_axis = np.argsort(pred)[::-1]
        sorted_pred = np.array(pred)[sort_axis]
        sorted_real = np.array(real)[sort_axis]
        sorted_loc = np.array(loc)[sort_axis]


        m = numberofLOC
        PMI = 0
        locsum = sum([sorted_loc[i] for i in range(0, m)])
        PLI = locsum / L

    tp = sum([1 if sorted_real[j] > 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
    fn = sum([1 if sorted_real[j] > 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
    fp = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
    tn = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
    # print('tp:{0},fn:{1},fp:{2},tn:{3}'.format(tp,fn,fp,tn))

    if (tp + fp == 0):
        Precision = 0
    else:
        Precision = tp / (tp + fp)

    if (tp + fn == 0):
        Recall = 0
    else:
        Recall = tp / (tp + fn)

    if (Recall + Precision == 0):
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    if (tp + fn + fp + tn == 0):
        Precisionx = 0
    else:
        Precisionx = (tp + fn) / (tp + fn + fp + tn)

    if (P == 0):
        Recallx = 0
    else:
        Recallx = (tp + fn) / P

    if (Recallx + Precisionx == 0):
        F1x = 0
    else:
        F1x = 2 * Recallx * Precisionx / (Recallx + Precisionx)

    if (fp + tn == 0):
        PF = 0
    else:
        PF = fp / (fp + tn)
    if (tp + fp == 0):
        falsealarmrate = 0
    else:
        falsealarmrate = fp / (tp + fp)

    PofB = sum([sorted_real[j] if sorted_real[j] > 0 else 0 for j in range(m)]) / Q

    PofBPMLI = PofB / PLI

    return Precisionx, Recallx, F1x, PF, falsealarmrate, PLI, PofB, PofBPMLI


def testModel(data_path, model_path, flag, logorlinear, target, savepath, modelname):
    csv_list = os.listdir(data_path)
    output_result = []
    for file in csv_list:
        fold_path = path + "/" + file
        print("Processing........ \n Filename: " + file)
        csv_list1 = os.listdir(fold_path)
        train_path = fold_path + '/' + csv_list1[0]
        test_path = fold_path + '/' + csv_list1[1]

        train_data = np.array(pd.read_csv(train_path))
        trainX = train_data[:, :-1]
        trainY = train_data[:, -1].astype(int)
        test_data = np.array(pd.read_csv(test_path))
        testX = test_data[:, :-1]
        testY = test_data[:, -1].astype(int)

        # nn-filter
        # trainX, trainY = NN_filter(trainX, trainY, testX)

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
            # Cla_testing_data_y = [1 if y > 0 else 0 for y in testY]
        testing_data_y = testY.tolist()
        temp_result = [0] * 16

        for r in range(10):

            modelsavepath = os.path.join(model_path, modelname + file + '_version_' + str(r) + '.pkl')

            model = joblib.load(modelsavepath)

            cost = train_loc
            if flag == 'number':
                training_data_y = trainY
            else:
                training_data_y = Cla_training_data_y
            if modelname == "LTR-linear":
                de = LTR(X=trainX, y=training_data_y, cost=cost, costflag='loc', logorlinear=logorlinear,
                         metircs=target)
                pred_prob = de.predict(testX, model)
            else:
                pred_prob = model.predict(testX)
            # fpa = PerformanceMeasure(testing_data_y, pred_prob).FPA()
            testingcodeN = test_loc
            Precision, Recall, F1, Precisionx, Recallx, F1x, PF20module, \
            falsealarmrate20module, IFMA, IFLA, PMI, PLI, PofB, PofBPLI \
                = PerformanceMeasure(
                testing_data_y, pred_prob, testingcodeN, 0.2, 'defect', 'module').getSomePerformance()

            Precisionx30, Recallx30, F1x30, PF30, falsealarmrate30, PLI30, PofB30, PofBPMI30 \
                = locPerformance(testing_data_y, pred_prob, testingcodeN, 30, 'defect', 'module')

            Precisionx60, Recallx60, F1x60, PF60, falsealarmrate60, PLI60, PofB60, PofBPMI60 \
                = locPerformance(testing_data_y, pred_prob, testingcodeN, 60, 'defect', 'module')

            # normpercentpopt20module = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, 0.2, 'defect',
            #                                              'module').PercentPOPT()
            Wholepopt = PerformanceMeasure(testing_data_y, pred_prob, testingcodeN, 1, 'defect', 'module').POPT()

            temp_result1 = [Precisionx, Recallx, F1x, PofB, PLI, PofBPLI, Wholepopt, IFMA,
                            F1x30, PofB30, PLI30, PofBPMI30,
                            F1x60, PofB60, PLI60, PofBPMI60]
            temp_result = np.sum([temp_result, temp_result1], axis=0).tolist()
        temp_result = [item / 10 for item in temp_result]
        output_result.append(temp_result)

    writeToFile(modelname, output_result, savepath)

def my_fpa_score(realbug, predbug):
    return PerformanceMeasure(realbug, predbug).FPA()


def writeToFile(name, list, output_path):
    header_name = ['Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PLI@20%', 'PofB/PLI@20%', 'Popt', 'IFA',
                   'F1@30', 'PofB@30', 'PLI@30', 'PofB/PLI@30',
                   'F1@60', 'PofB@60', 'PLI@60', 'PofB/PLI@60']

    dir = output_path
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = dir + name + '.csv'
    df = pd.DataFrame(list, columns=header_name)
    df.to_csv(path, index=False)


if __name__ == '__main__':

    flags = ['number', 'without_number']
    path = '../crossrelease_csv'
    models = ["RFR", "RR", "LTR-linear"]
    for flag in flags:
        for model in models:
            logorlinear = 'linear'
            target = 'pofb20'
            model_path = '../Models_count/CrossRelease/' + flag + '/' + model + '/'
            save_path = '../output_Count_final/CrossRelease/' + flag + '/'
            print('Save to ...' + save_path)
            testModel(path, model_path, flag, logorlinear, target, save_path, model)


