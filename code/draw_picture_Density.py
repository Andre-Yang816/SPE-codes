import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_color(path, func, metrics):
    colors_path = '../Picture_final/Wilcoxon/Density/{0}/p_{1}.csv'.format(path, func)
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
    new_metrics = ['Precision@20%LOC', 'Recall@20%LOC', 'F1@20%LOC', 'PofB@20%LOC', 'PMI@20%LOC', 'PofB/PMI@20%LOC', 'Popt@LOC',
               'F1@3000LOC', 'PofB@3000LOC', 'PMI@3000LOC', 'PofB/PMI@3000LOC',
               'F1@5000LOC', 'PofB@5000LOC', 'PMI@5000LOC', 'PofB/PMI@5000LOC']
    plt.xticks(xticks, new_metrics, rotation=90, weight='heavy', fontsize=12, ha='center')
    plt.yticks(fontsize=12, weight='heavy')
    plt.ylabel(func, fontsize=12, weight='heavy')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.axhline(y=0, color='blue', lw=1)

    # plt.axvline(6.3, color='grey', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    '''
    plt.axvline(18, color='grey', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    plt.axvline(34, color='grey', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标
    # plt.axvline(36.5, color='black', linestyle=':')  # 添加一个竖线，4.5表示竖线的x轴坐标

    plt.title(
        "     Data filter    "
        "    Transfer learning"
        , fontsize=14, loc='left', weight='heavy')
    '''
    Path('../Picture_final/Density/{0}'.format(path)).mkdir(parents=True, exist_ok=True)
    if path == 'crossRelease':
        output_path = '../Picture_final/Density/{0}/density_cr_{1}.pdf'.format(path, func)
    else:
        output_path = '../Picture_final/Density/{0}/density_cp_{1}.pdf'.format(path, func)

    foo_fig = plt.gcf()
    foo_fig.savefig(output_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.clf()
    plt.close()


def saveImprovement(data, path, func):

    metrics = ['Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PMI@20%', 'PofB/PMI@20%', 'Popt',
               'F1@3000', 'PofB@3000', 'PMI@3000', 'PofB/PMI@3000',
               'F1@5000', 'PofB@5000', 'PMI@5000', 'PofB/PMI@5000']

    dir = '../Picture_final/Improvement/density/'+path
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = dir +'/'+ func +'.csv'
    df = pd.DataFrame(data).T
    df.to_csv(path, index=False, header=metrics)


if __name__ == '__main__':
    functions = ['LTR-linear', 'LTR-log', 'DEJIT']
    data_path = '../output_Density_final/'
    metrics = ['Precision@20%', 'Recall@20%', 'F1@20%', 'PofB@20%', 'PMI@20%',  'PofB/PMI@20%', 'Popt',
                   'F1@3000', 'PofB@3000', 'PMI@3000', 'PofB/PMI@3000',
                   'F1@5000', 'PofB@5000', 'PMI@5000', 'PofB/PMI@5000']
    paths = ['crossRelease', 'crossProject']
    ifa = 'IFA'
    for path in paths:
        for func in functions:
            number_data_path = data_path + path + '/number'
            without_number_data_path = data_path + path + '/without_number'
            number_data = read_data(number_data_path, func)
            without_number_data = read_data(without_number_data_path, func)
            colors = load_color(path, func, metrics)
            improvement = processDatas(number_data, without_number_data)
            saveImprovement(improvement, path, func)

            drawFigure(improvement, metrics, func, colors, path)