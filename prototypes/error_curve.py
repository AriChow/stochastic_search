'''
This file is for showing the error curves of algorithms w.r.t 
a dataset on validation error.
'''

import pickle
import os
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
from prototypes.data_analytic_pipeline import image_classification_pipeline



home = os.path.expanduser('~')
data_name = 'breast'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/stochastic_search/results/'
pipeline = {}
steps = ['feature_extraction', 'dimensionality_reduction', 'learning_algorithm']
pipeline['feature_extraction'] = ["haralick", "VGG", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["RF", "SVM"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']
run = 4

# Random
type1 = 'random_MCMC'
random_val_error = []
obj = pickle.load(open(results_home + 'intermediate/random/' + type1 + '_' + data_name + '_run_' +
					   str(run) + '_full.pkl', 'rb'), encoding='latin1')
times = np.asarray(obj.times)
times -= times[0]
times = times.tolist()
v = []
for i in range(len(times)):
	v.append((obj.error_curve[i], times[i]))
random = np.zeros((len(v), 2))
for i, v1 in enumerate(v):
	random[i, 0] = v1[0]
	random[i, 1] = v1[1]

# Stochastic search
type1 = 'RL_MCMC'
obj = pickle.load(open(results_home + 'intermediate/SS/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl', 'rb'), encoding='latin1')

times = np.asarray(obj.times)
times -= times[0]
times = times.tolist()
v = []
for i in range(len(times)):
	v.append((obj.error_curve[i], times[i]))
SS = np.zeros((len(v), 2))
for i, v1 in enumerate(v):
	SS[i, 0] = v1[0]
	SS[i, 1] = v1[1]

# SMAC
type1 = 'SMAC'
obj = pickle.load(open(results_home + 'intermediate/SMAC/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full_naive.pkl', 'rb'), encoding='latin1')

times = np.asarray(obj.times)
times -= times[0]
times = times.tolist()
v = []
for i in range(len(times)):
	v.append((obj.error_curves[0][i], times[i]))
SMAC = np.zeros((len(v), 2))
for i, v1 in enumerate(v):
	SMAC[i, 0] = v1[0]
	SMAC[i, 1] = v1[1]


plt.plot(random[:, 1], random[:, 0], color='r', label='random search')
plt.plot(SMAC[:, 1], SMAC[:, 0], color='g', label='Bayesian optimization')
plt.plot(SS[:, 1], SS[:, 0], color='b', label='Stochastic search')

plt.title('Validation error curves')
plt.xlabel('Time (s)')
plt.ylabel('Validation error')
plt.legend()
plt.savefig(results_home + 'figures/validation_curve_iter_1_' + data_name + '.jpg')
plt.close()
