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
    supervised_20 = [.2, .1866, .6, .266, .5, .233, .233, .333, .2, .266]
    supervised_40 = [.433, .233, .733, .433, .4, .333, .466, .4, .466, .366]
    supervised_60 = [.6, .633, .733, .633, .6, .533, .7, .533, .633, .566]
    DAgger_20 = [.433, .233, .366, .466, .1, .433, .433, .3, .466, .433]
    DAgger_40 = [.566, .366, .366, .533, .3, .333, .233, .6, .366, .366]
    
    aggregated_sup = [supervised_20, supervised_40, supervised_60]
    aggregated_dag = [supervised_20, DAgger_20, DAgger_40]
    sup_means = map(np.mean, aggregated_sup)
    dag_means = map(np.mean, aggregated_dag)

    sup_err = map(compute_std_er_m, aggregated_sup)
    dag_err = map(compute_std_er_m, aggregated_dag)

    amnts = [0, 20, 40, 60]
    # plt.errorbar(amnts,sup_means,yerr=sup_err,linewidth=3.0, color='r', label='supervised')
    # plt.errorbar(amnts,dag_means,yerr=dag_err,linewidth=3.0, color='b', label='DAgger')

    print sup_means
    print dag_means
    print sup_err
    print dag_err
    fig, ax = plt.subplots()
    plt.errorbar(amnts, [0]+dag_means, [0]+dag_err, linewidth=2.0, color='orange', marker='o', ecolor='black', elinewidth=1.0, markeredgecolor='orange', markeredgewidth=2.5, markerfacecolor='white')
    plt.errorbar(amnts, [0]+sup_means, [0]+sup_err, linewidth=2.0, color='blue', marker='o', ecolor='black', elinewidth=1.0, markeredgecolor='blue', markeredgewidth=2.5, markerfacecolor='white')

    # ax.errorbar(amnts,sup_means,yerr=sup_err,linewidth=3.0, color='r', label='Supervised')
    # ax.errorbar(amnts,dag_means,yerr=dag_err,linewidth=3.0, color='b', label='DAgger')
    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper center', shadow=True)


    plt.title('Test Rates for Human Experiments')
    plt.xlabel('Number of Rollouts in Training Set')
    plt.ylabel('Probability of Success on Test Set')
    plt.show()