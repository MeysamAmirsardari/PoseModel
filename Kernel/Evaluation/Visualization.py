
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pickle

### From DeepLabCut:
# with open('ResultsCorruption/data_corruption_experiments.pickle', 'rb') as file:
#     data = pickle.load(file)
#
# bpts = ['snout', 'leftear', 'rightear', 'tailbase']
# n_bpts = len(bpts)
# mags = sorted(data['Magnitude'].unique())
# test_error = data[~data['Training set']]
#
# cmap = plt.cm.get_cmap('Blues', 7)
# colors = cmap(range(2, 7))
#
# for frac in sorted(test_error['Training size'].unique()):
#
#     fig, axes = plt.subplots(2, 4, figsize=(5.32, 3.14), dpi=100)
#     for i, mag in enumerate(mags):
#         d = test_error[((test_error['Training size'] == frac) &
#                         (test_error['Magnitude'] == mag))]
#         for (bpt, corrupt), temp in d.groupby(['Bodyparts', 'Corruption']):
#             n = bpts.index(bpt)
#             if corrupt == 'swap':
#                 n += 4
#             sorted_errors = np.sort(temp['Error'])
#             frac_invalid = (sorted_errors == 0).sum() / len(sorted_errors)
#             sorted_errors = sorted_errors[sorted_errors != 0]
#             x = np.concatenate([sorted_errors, sorted_errors[[-1]]])
#             y = np.linspace(0, 1 - frac_invalid, len(x))
#             axes.flat[n].step(x, y, label=mag, color=colors[i], lw=2, where='mid')
#             if corrupt!='swap':
#                 axes.flat[n].set_title(bpt)
#
#             if corrupt == 'default':
#                 axes.flat[n + 4].step(x, y, label=mag, color=colors[i], lw=2)
#     for n, ax in enumerate(axes.flat):
#         ax.tick_params(axis='both', direction='in')
#
#         # ax.set_box_aspect(1)
#         # For matplotlib < 3.3.0, uncomment the line below:
#         #ax.set_aspect(20, adjustable='datalim')
#         ax.set_xlim(0, 20)
#         ax.set_ylim(0, 1)
#         ax.spines['left'].set_position(('outward', 10))
#         ax.spines['bottom'].set_position(('outward', 10))
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         if n < 4:
#             ax.set_xticks([])
#             ax.set_xticklabels([])
#             ax.spines['bottom'].set_visible(False)
#         if n % 4 != 0:
#             ax.spines['left'].set_visible(False)
#             ax.set_yticks([])
#             ax.set_yticklabels([])
#     patches = [plt.plot([0, 0], [0, 0], color=color, label=f'{label}%', lw=2)[0]
#                for color, label in zip(colors, mags)]
#
#     fig.legend(handles=patches, loc='center', frameon=False,
#                bbox_to_anchor=(0, 0.45, 1, 0.1),
#                ncol=5, borderaxespad=0.)
#     plt.savefig('ResultsCorruption/LabelingCorruptionimpact_'+str(frac)+'Percent.png')
#
# def compare(old, new):
#     assert len(old) == len(new)
#
#     idx = np.linspace(0, len(old), len(old))
#     #fig, ax = plt.subplots()
#     plt.plot(idx, old-new)
#     plt.show()
#     # ax.plot(idx, new, label='Old')
#     # ax.set_xlabel('x label')  # Add an x-label to the axes.
#     # ax.set_ylabel('y label')  # Add a y-label to the axes.
#     # ax.set_title("Simple Plot")  # Add a title to the axes.
#     # ax.legend()  # Add a legend.
#     pass

