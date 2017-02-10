import sys
sys.path.append('/home/annal/Izzy/vision_amt/scripts/objects/')
import matplotlib.patches as mpatches
# from helper import Rectangle, sign, cosine, sine, w, h, getRot, pasteOn
# from PIL import Image
import numpy as np
import math
import random
import matplotlib.pyplot as plt
# from pipeline.bincam import BinaryCamera
import sys, os, time, cv2
# from Net.tensor import inputdata
# from options import AMTOptions
def compute_std_er_m(data):
    n = len(data)
    std = np.std(data)
    print std
    return std/np.sqrt(n)


if __name__ == '__main__':
    # supervised_20 = [.2, .1866, .6, .266, .5, .233, .233, .333, .2, .266]
    # supervised_40 = [.433, .233, .733, .433, .4, .333, .466, .4, .466, .366]
    # supervised_60 = [.6, .633, .733, .633, .6, .533, .7, .533, .633, .566]
    # DAgger_20 = [.433, .233, .366, .466, .1, .433, .433, .3, .466, .433]
    # DAgger_40 = [.566, .366, .366, .533, .3, .333, .233, .6, .366, .366]
    
    # aggregated_sup = [supervised_20, supervised_40, supervised_60]
    # aggregated_dag = [supervised_20, DAgger_20, DAgger_40]
    # sup_means = map(np.mean, aggregated_sup)
    # dag_means = map(np.mean, aggregated_dag)

    # sup_err = map(compute_std_er_m, aggregated_sup)
    # dag_err = map(compute_std_er_m, aggregated_dag)

    # amnts = [0, 20, 40, 60]
    # # plt.errorbar(amnts,sup_means,yerr=sup_err,linewidth=3.0, color='r', label='supervised')
    # # plt.errorbar(amnts,dag_means,yerr=dag_err,linewidth=3.0, color='b', label='DAgger')

    # print sup_means
    # print dag_means
    # print sup_err
    # print dag_err
    # fig, ax = plt.subplots()
    # plt.errorbar(amnts, [0]+dag_means, [0]+dag_err, linewidth=2.0, color='orange', marker='o', ecolor='black', elinewidth=1.0, markeredgecolor='orange', markeredgewidth=2.5, markerfacecolor='white')
    # plt.errorbar(amnts, [0]+sup_means, [0]+sup_err, linewidth=2.0, color='blue', marker='o', ecolor='black', elinewidth=1.0, markeredgecolor='blue', markeredgewidth=2.5, markerfacecolor='white')

    # # ax.errorbar(amnts,sup_means,yerr=sup_err,linewidth=3.0, color='r', label='Supervised')
    # # ax.errorbar(amnts,dag_means,yerr=dag_err,linewidth=3.0, color='b', label='DAgger')
    # # Now add the legend with some customizations.
    # legend = ax.legend(loc='upper center', shadow=True)


    # plt.title('Test Rates for Human Experiments')
    # plt.xlabel('Number of Rollouts in Training Set')
    # plt.ylabel('Probability of Success on Test Set')
    # plt.show()

    # N = 5
    # improvement_rates = (5.8175, 7.4998825, 11.666655, -1.666666667, 1.666666575)
    # error_std = (7.382313887/2, 6.30994292/2,6.382847965/2, 5.773502692/2, 4.303314747/2)
    # ind = np.arange(N)    # the x locations for the groups
    # width = 0.35       # the width of the bars: can also be len(x) sequence

    # p1 = plt.bar(ind, improvement_rates, width, color='r', yerr=error_std)
    # # p2 = plt.bar(ind, womenMeans, width, color='y',
    # #              bottom=menMeans, yerr=womenStd)

    # plt.ylabel('Percentage Improvement')
    # plt.title('Improvement by applying noise correction')
    # plt.xticks(ind + width/2., ('Adaptive', 'Smoothing', 'Combined', 'Removed all', 'Removed human'))
    # plt.yticks(np.arange(-10, 25, 5))
    # # plt.legend((p1[0], p2[0]), ('Men', 'Women'))

    # plt.show()

    # N = 6
    # improvement_rates = (0.0776665, 0.074998825, 0.1124999663, -0.01666666667, 0.01666666575, -0.008333335)
    # improvement_rates = np.array(improvement_rates) * 100
    # error_std = (0.04306462356, 0.02065401543, 0.02083333261, 0.02886751346, 0.02151657374, 0.03435921233)
    # error_std = np.array(error_std) * 100
    # ind = np.arange(N)    # the x locations for the groups
    # width = 0.5       # the width of the bars: can also be len(x) sequence

    # p1 = plt.bar(ind, improvement_rates, width, color='r', yerr=error_std)
    # # p2 = plt.bar(ind, womenMeans, width, color='y',
    # #              bottom=menMeans, yerr=womenStd)

    # plt.ylabel('Percentage Improvement')
    # plt.title('Improvement by applying noise correction')
    # plt.xticks(ind + width/2., ('Adaptive', 'Smoothing', 'Combined', 'Remove', 'Cut', 'Robot'))
    # plt.yticks(np.arange(-10, 25, 5))
    # # plt.legend((p1[0], p2[0]), ('Baseline', 'Improvement'))

    # # plt.legend((p1[0], p2[0]), ('Baseline', 'Improvement'))

    # plt.show()


    N = 4
    improvement_rates = (0.0776665, -0.01666666667, 0.01666666575, -0.008333335)
    improvement_rates = np.array(improvement_rates) * 100
    error_std = (0.04306462356, 0.02886751346, 0.02151657374, 0.03435921233)
    error_std = np.array(error_std) * 100
    ind = np.arange(N)    # the x locations for the groups
    width = 0.5       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, improvement_rates, width, color='r', yerr=error_std)
    # p2 = plt.bar(ind, womenMeans, width, color='y',
    #              bottom=menMeans, yerr=womenStd)

    plt.ylabel('Percentage Improvement')
    plt.title('Improvement by Applying Error Correction Techniques')
    plt.xticks(ind + width/2., ('Correction', 'Remove', 'Cut', 'Robot'))
    plt.yticks(np.arange(-10, 25, 5))
    # plt.legend((p1[0], p2[0]), ('Baseline', 'Improvement'))

    # plt.legend((p1[0], p2[0]), ('Baseline', 'Improvement'))

    plt.show()

    N = 3
    improvement_rates = (0.0776665, 0.074998825, 0.1124999663)
    improvement_rates = np.array(improvement_rates) * 100
    error_std = (0.04306462356, 0.02065401543, 0.02083333261)
    error_std = np.array(error_std) * 100
    ind = np.arange(N)    # the x locations for the groups
    width = 0.5       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, improvement_rates, width, color='r', yerr=error_std)
    # p2 = plt.bar(ind, womenMeans, width, color='y',
    #              bottom=menMeans, yerr=womenStd)

    plt.ylabel('Percentage Improvement')
    plt.title('Improvement by applying noise correction')
    plt.xticks(ind + width/2., ('Correction', 'Smoothing', 'Combined'))
    plt.yticks(np.arange(-10, 25, 5))
    # plt.legend((p1[0], p2[0]), ('Baseline', 'Improvement'))

    # plt.legend((p1[0], p2[0]), ('Baseline', 'Improvement'))

    plt.show()