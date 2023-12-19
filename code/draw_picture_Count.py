import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
def load_color(path, func, metrics):
    colors_path = '../Picture_final/Wilcoxon/{0}/p_{1}.csv'.format(path, func)
    datas = pd.read_csv(colors_path)

    colors = []
    for me in metrics:
        if datas[me][0] < 0.05:
            colors.append('red')
        else:
            colors.append('black')

    return colors


def read_data(dataset_path, func):

    datas = {}
    for me in metrics:
        data_path = '{0}/{1}.csv'.format(dataset_path,func)
        raw_datas = pd.read_csv(data_path)

        raw_datas = raw_datas[me].values

        datas[me] = raw_datas

    return datas

def processDatas(dataA, dataB):
    improvement = []
    for me in metrics:
        improvement.append(dataA[me]-dataB[me])
    return improvement

def drawFigure(metric_datas, metrics, func, colors, path):
    ymax = 0
    ymin = 100
    for data in metric_datas:
        if ymax < max(data):
            ymax = max(data)
        if ymin > min(data):
            ymin = min(data)

    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.tick_params(direction='in')

    xticks = np.arange(1, len(metrics) * 1.5, 1.5)
    figure = ax.boxplot(metric_datas,
                        notch=False,  # notch shape
                        sym='r+',  # blue squares for outliers
                        vert=True,  # vertical box aligmnent
                        meanline=True,
                        showmeans=False,
                        patch_artist=False,
                        showfliers=False,
                        positions=xticks,
                        boxprops={'color': 'red'}
                        )

    for i in range(len(colors)):
        k = figure['boxes'][i]
        k.set(color=colors[i])
        k = figure['medians'][i]
        k.set(color=colors[i], linewidth=2)
        k = figure['whiskers'][2 * i:2 * i + 2]
        for w in k:
            w.set(color=colors[i], linestyle='--')
        k = figure['caps'][2 * i:2 * i + 2]
        for w in k:
            w.set(color=colors[i])
    #plt.xlim((0, 10))
    # metrics_new = []
    # for func in metrics:
    #    metrics_new.append(func[:-4])
    new_metrics = ['Precision@20%module', 'Recall@20%module', 'F1@20%module', 'PofB@20%module', 'PLI@20%module',  'PofB/PLI@20%module', 'Popt@module',
               'F1@30module', 'PofB@30module', 'PLI@30module',  'PofB/PLI@30module',
               'F1@60module', 'PofB@60module', 'PLI@60module',  'PofB/PLI@60module']
    plt.xticks(xticks, new_metrics, rotation=90, weight='heavy', fontsize=12, ha='center')
    plt.yticks(fontsize=12, weight='heavy')
    plt.ylabel(func, fontsize=12, weight='heavy')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.axhline(y=0, color='blue', lw=1)

    # plt.axvline(6.3, color='grey', linestyle=':')

    Path('../Picture_final/count/{0}'.format(path)).mkdir(parents=True, exist_ok=True)
    if path == 'crossRelease':
        output_path = '../Picture_final/count/{0}/count_cr_{1}.pdf'.format(path, func)
    else:
        output_path = '../Picture_final/count/{0}/count_cp_{1}.pdf'.format(path, func)
    foo_fig = plt.gcf()
    foo_fig.savefig(output_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.clf()
    plt.close()

def saveImprovement(data, path, func):

    metrics = ['Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PLI@20%',  'PofB/PLI@20%', 'Popt',
               'F1@30', 'PofB@30', 'PLI@30',  'PofB/PLI@30',
               'F1@60', 'PofB@60', 'PLI@60',  'PofB/PLI@60']

    dir = '../Picture_final/Improvement/count/'+path
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = dir + '/' + func +'.csv'
    df = pd.DataFrame(data).T
    df.to_csv(path, index=False, header=metrics)

if __name__ == '__main__':
    functions1 = ['RFR', 'RR', 'LTR-linear']
    functions2 = ['LR', 'RR', 'LTR-linear']
    data_path = '../output_Count_final/'
    metrics = ['Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PLI@20%',  'PofB/PLI@20%', 'Popt',
               'F1@30', 'PofB@30', 'PLI@30',  'PofB/PLI@30',
               'F1@60', 'PofB@60', 'PLI@60',  'PofB/PLI@60']
    paths = ['crossProject', 'crossRelease']
    ifa = 'IFA'
    for path in paths:
        if path == 'crossRelease':
            functions = functions1
        else:
            functions = functions2
        for func in functions:
            number_data_path = data_path + path + '/number'
            without_number_data_path = data_path + path + '/without_number'
            number_data = read_data(number_data_path, func)
            without_number_data = read_data(without_number_data_path, func)
            colors = load_color(path, func, metrics)
            improvement = processDatas(number_data, without_number_data)
            saveImprovement(improvement, path, func)
            drawFigure(improvement, metrics, func, colors, path)





