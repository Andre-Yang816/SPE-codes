import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def load_color(path, func, metrics):
    colors_path = '../Picture_final/Wilcoxon/{0}/p_{1}.csv'.format(path, func)
    datas = pd.read_csv(colors_path)

    colors = 'black'
    if datas[metrics][0] < 0.05:
        colors='red'

    return colors


def read_data(dataset_path, func, metrics):

    datas = {}

    data_path = '{0}/{1}.csv'.format(dataset_path,func)
    raw_datas = pd.read_csv(data_path)

    raw_datas = raw_datas[metrics].values

    datas[metrics] = raw_datas

    return datas

def processDatas(dataA, dataB, metric):
    improvement = dataA[metric]-dataB[metric]
    return improvement

def drawFigure(metric_datas, metrics, funcs, colors, path):

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

    xticks = np.arange(1, len(funcs) * 1.5, 1.5)
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
    plt.xticks(xticks, funcs, rotation=35, weight='heavy', fontsize=12, ha='center')
    plt.yticks(fontsize=12, weight='heavy')
    plt.ylabel(metrics, fontsize=12, weight='heavy')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.axhline(y=0, color='blue', lw=1)

    plt.axvline(4.7, color='grey', linestyle=':')

    Path(path).mkdir(parents=True, exist_ok=True)
    output_path = '{0}/count_{1}.pdf'.format(path, metrics)
    foo_fig = plt.gcf()
    foo_fig.savefig(output_path, format='pdf', dpi=1000, bbox_inches='tight')
    plt.clf()
    plt.close()


if __name__ == '__main__':
    functions1 = ['RFR', 'RR', 'LTR-linear']
    functions2 = ['LR', 'RR', 'LTR-linear']
    data_path = '../output_Count_final/'
    paths = ['crossRelease', 'crossProject']
    ifa = 'IFA'

    colors_ifa = []
    improvement_ifa = []
    for path in paths:
        if path == 'crossRelease':
            functions = functions1
        else:
            functions = functions2
        for func in functions:
            print(func)
            number_data_path = data_path + path + '/number'
            without_number_data_path = data_path + path + '/without_number'
            number_data = read_data(number_data_path, func, ifa)
            without_number_data = read_data(without_number_data_path, func, ifa)
            colors = load_color(path, func, ifa)
            colors_ifa.append(colors)
            improvement = processDatas(number_data, without_number_data, ifa)
            improvement_ifa.append(improvement)

    x_label = ['RFR(cross-release)', 'RR(cross-release)', 'LTR-linear(cross-release)',
               'LR(cross-project)', 'RR(cross-project)', 'LTR-linear(cross-project)']
    path_ifa = '../Picture_final/count/'
    drawFigure(improvement_ifa, ifa, x_label, colors_ifa, path_ifa)





