import numpy as np
import matplotlib.pyplot as plt

mean_opt_2 = 14.1213
std_opt_2 = 0.7911
mean_opt_3 = 20.04
std_opt_3 = 1.159
mean_opt_4 = 19.39
std_opt_4 = 4.075

mean_const_2 = 9.19
std_const_2 = 0.95
mean_const_3 = 9.31
std_const_3 = 1.66
mean_const_4 = 8.91
std_const_4 = 5.34


means = [19.61056501, 14.06749313]
stdevs = [1.927156765, 1.358788555]
exes = [0, 1]
col_labs = ['Optimized Speed', 'Constant Speed']

ol_locs = [2]
fig, ax = plt.subplots(layout='constrained')
ax.scatter(col_locs[0], means[0], c='c', linestyle='None', linewidths = 5, label='Optimized Speed')
ax.errorbar(col_locs[0], means[0], c='c', linestyle='None', yerr=stdevs[0], capsize=10, elinewidth=3)


ax.scatter(col_locs[0], means[1], c='m', linestyle='None', linewidths = 5,label='Constant Speed')
ax.errorbar(col_locs[0], means[1], c='m', linestyle='None', yerr=stdevs[1], capsize=10, elinewidth=3)


ax.set_ylabel('Profit Rate (USD/hour)')
ax.set_xticks(col_locs)
ax.set_xticklabels(col_labs)
ax.set_title('Physical Experiment:\n2 material sorting')
ax.yaxis.grid(True)
ax.legend(loc='lower left')
ax.set_ylim(0, 25)
ax.set_xlim(1, 3)
plt.show()
