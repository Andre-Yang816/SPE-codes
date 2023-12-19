#input:number/ltr-linear,without_number/ltr-linear
#step:
# 1. Calculate Wilcox
# 2. Calculate the difference
# 3. Draw a picture

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def read_data1(dataset_path, func):
    data_path = '{0}/{1}.csv'.format(dataset_path, func)
    raw_datas = pd.read_csv(data_path)
    return raw_datas

def read_data(dataset_path, func):

    datas = {}
    for me in metrics:
        data_path = '{0}/{1}.csv'.format(dataset_path,func)
        raw_datas = pd.read_csv(data_path)

        raw_datas = raw_datas[me].values

        datas[me] = raw_datas

    return datas

def wilcoxon(l1, l2):
    w, p_value = stats.wilcoxon(l1, l2, correction=False)
    return p_value


if __name__ == '__main__':
    functions1 = ['RFR', 'RR', 'LTR-linear']
    functions2 = ['LTR-linear', 'LR', 'RR']
    data_path = '../output_Count_final/'
    paths = ['CrossProject', 'CrossRelease']

    metrics = ['Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PLI@20%',  'PofB/PLI@20%', 'Popt', 'IFA',
               'F1@30', 'PofB@30', 'PLI@30',  'PofB/PLI@30',
               'F1@60', 'PofB@60', 'PLI@60',  'PofB/PLI@60']

    for path in paths:
        if path == 'CrossRelease':
            functions = functions1
        else:
            functions = functions2
        for func in functions:
            number_data_path = data_path + path + '/number'
            without_number_data_path = data_path + path + '/without_number'
            number_data = read_data(number_data_path, func)
            without_number_data = read_data(without_number_data_path, func)
            #number_data = np.array(number_data)
            #without_number_data = np.array(without_number_data)
            # print(number_data_path)
            # print(without_number_data)
            pvalues = []
            bhpvalues = []
            sortpvalues = []
            for m in metrics:
                pvalue = wilcoxon(number_data[m], without_number_data[m])
                pvalues.append(pvalue)
                sortpvalues.append(pvalue)
            sortpvalues.sort()

            for i in range(len(pvalues)):
                bhpvalue = pvalues[i] * (len(pvalues)) / (sortpvalues.index(pvalues[i]) + 1)
                bhpvalues.append(bhpvalue)
                print("compute Benjamini—Hochberg p-value between %s number and %s without_number: %s" % (
                func, func, bhpvalue))
            Path('../Picture_final/Wilcoxon/{0}'.format(path)).mkdir(parents=True, exist_ok=True)
            output_path = '../Picture_final/Wilcoxon/{0}/p_{1}.csv'.format(path, func)

            output = pd.DataFrame(data=[pvalues], columns=metrics)
            # output = pd.DataFrame(data=[pvalues], columns=functions)
            output.to_csv(output_path, encoding='utf-8')





