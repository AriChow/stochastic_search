'''
This file is for showing the comparative ranking of algorithms w.r.t 
to datasets on test error and validation error.
'''

import pickle
import os
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
from prototypes.data_analytic_pipeline import image_classification_pipeline

home = os.path.expanduser('~')
datasets = ['breast', 'matsc_dataset1', 'matsc_dataset2', 'brain']
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/stochastic_search/results/'
pipeline = {}
steps = ['feature_extraction', 'dimensionality_reduction', 'learning_algorithm']
pipeline['feature_extraction'] = ["haralick", "VGG", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["RF", "SVM"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']


# Random
start = 1
stop = 6
type1 = 'random_MCMC'
random_val_error = []
random_test_error = []

# Output is 2 5x5 matrices (val, test) with rows representing datasets
# and columns representing runs
for id, data_name in enumerate(datasets):
	val = []
	test = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/random/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl', 'rb'), encoding='latin1')

		# Find minimum validation error
		min_err = 100000000
		pipelines = obj.pipelines
		best_pipeline = []
		for p in pipelines:
			e = p.get_error()
			if e < min_err:
				min_err = e
				best_pipeline = p
		val.append(min_err)

		# # Find minimum test error
		# best_pipeline.data_location = data_home
		# best_pipeline.feature_extraction = best_pipeline.feature_extraction.decode('latin1')
		# best_pipeline.dimensionality_reduction = best_pipeline.dimensionality_reduction.decode('latin1')
		# best_pipeline.learning_algorithm = best_pipeline.learning_algorithm.decode('latin1')
		# best_pipeline.ml_type = 'testing'
		# best_pipeline.run()
		# test.append(best_pipeline.get_error())

	val = np.expand_dims(np.asarray(val), 0)
	test = np.expand_dims(np.asarray(test), 0)
	if id == 0:
		random_val_error = val
		random_test_error = test
	else:
		random_val_error = np.vstack((random_val_error, val))
		random_test_error = np.vstack((random_test_error, test))



# TPE
start = 1
stop = 4
type1 = 'TPE'
TPE_val_error = []
TPE_test_error = []

# Output is 2 5x5 matrices (val, test) with rows representing datasets
# and columns representing runs
for id, data_name in enumerate(datasets):
	val = []
	test = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/TPE/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl', 'rb'), encoding='latin1')

		# Find minimum validation error
		trials = obj[1]
		val.append(trials.best_trial['result']['loss'])

		# # Find minimum test error
		# best_pipeline = obj[0]
		# fe = pipeline['feature_extraction'][best_pipeline['feature_extraction']]
		# dr = pipeline['dimensionality_reduction'][best_pipeline['dimensionality_reduction']]
		# la = pipeline['learning_algorithm'][best_pipeline['learning_algorithm']]
		# hyper = {}
		# for k in best_pipeline.keys():
		# 	if k not in steps:
		# 		hyper[k] = best_pipeline[k]
		# g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
		# 								  data_loc=data_home, type1=type1, fe=fe, dr=dr,
		# 								  la=la, val_splits=3, test_size=0.2)
		# g.run()
		# test.append(g.get_error())




	val = np.expand_dims(np.asarray(val), 0)
	test = np.expand_dims(np.asarray(test), 0)
	if id == 0:
		TPE_val_error = val
		TPE_test_error = test
	else:
		TPE_val_error = np.vstack((TPE_val_error, val))
		TPE_test_error = np.vstack((TPE_test_error, test))




# SMAC
start = 1
stop = 6
type1 = 'bayesian_MCMC'
SMAC_val_error = []
SMAC_test_error = []

# Output is 2 5x5 matrices (val, test) with rows representing datasets
# and columns representing runs
for id, data_name in enumerate(datasets):
	val = []
	test = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/SMAC/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl', 'rb'), encoding='latin1')

		# Find minimum validation error
		val.append(obj.error_curves[-1])

		# # Find minimum test error
		# best_pipeline = obj.best_pipelines[-1]._values
		# fe = best_pipeline['feature_extraction']
		# dr = best_pipeline['dimensionality_reduction']
		# la = best_pipeline['learning_algorithm']

		# hyper = {}
		# for k in best_pipeline.keys():
		# 	if k not in steps:
		# 		k1 = copy.deepcopy(k)
		# 		if 'isomap_' in k:
		# 			k1 = k[7:]
		# 		elif 'rf_' in k:
		# 			k1 = k[3:]
		# 		hyper[k1] = best_pipeline[k]
		# g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
		# 								data_loc=data_home, type1=type1, fe=fe, dr=dr,
		# 								la=la, val_splits=3, test_size=0.2)
		# g.run()
		# test.append(g.get_error())

	val = np.expand_dims(np.asarray(val), 0)
	test = np.expand_dims(np.asarray(test), 0)
	if id == 0:
		SMAC_val_error = val
		SMAC_test_error = test
	else:
		SMAC_val_error = np.vstack((SMAC_val_error, val))
		SMAC_test_error = np.vstack((SMAC_test_error, test))






# Stochastic Search
start = 1
stop = 6
type1 = 'RL_MCMC'
SS_val_error = []
SS_test_error = []

# Output is 2 5x5 matrices (val, test) with rows representing datasets
# and columns representing runs
for id, data_name in enumerate(datasets):
	val = []
	test = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/SS/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full_multivariate.pkl', 'rb'), encoding='latin1')

		# Find minimum validation error
		min_err = 100000000
		pipelines = obj.pipelines
		best_pipeline = []
		for p in pipelines:
			e = p[0].get_error()
			if e < min_err:
				min_err = e
				best_pipeline = p[0]
		val.append(min_err)

		# # Find minimum test error
		# best_pipeline.ml_type = 'testing'
		# best_pipeline.run()
		# test.append(best_pipeline.get_error())

	val = np.expand_dims(np.asarray(val), 0)
	test = np.expand_dims(np.asarray(test), 0)
	if id == 0:
		SS_val_error = val
		SS_test_error = test
	else:
		SS_val_error = np.vstack((SS_val_error, val))
		SS_test_error = np.vstack((SS_test_error, test))







random_val_mean = np.mean(random_val_error, axis=1)
random_val_std = np.std(random_val_error, axis=1)
# SMAC_val_mean = np.mean(SMAC_val_error, axis=1)
# SMAC_val_std = np.std(SMAC_val_error, axis=1)
SS_val_mean = np.mean(SS_val_error, axis=1)
SS_val_std = np.std(SS_val_error, axis=1)
TPE_val_mean = np.mean(TPE_val_error, axis=1)
TPE_val_std = np.std(TPE_val_error, axis=1)

colors = ['b', 'g', 'y', 'r']
algs = ['random', 'bayesian', 'SS']
x1 = np.asarray([1, 2, 3, 4])
w = 0.2
d = w * np.ones(4)
x2 = x1 - d
x3 = x1 + d
x4 = x3 + d
plt.bar(x2, random_val_mean.ravel(), width=w, align='center', color=colors[0], yerr=random_val_std.ravel(), label='Random')
#plt.bar(x1, SMAC_val_mean.ravel(), width=w, align='center', color=colors[0], yerr=SMAC_val_std.ravel(), label='SMAC' )
plt.bar(x1, TPE_val_mean.ravel(), width=w, align='center', color=colors[1], yerr=TPE_val_std.ravel(), label='bayesian (TPE)')
plt.bar(x3, SS_val_mean.ravel(), width=w, align='center', color=colors[2], yerr=SS_val_std.ravel(), label='Stochastic search')
plt.title('Ranking of algorithms in terms of validation error')
plt.xlabel('Datasets')
plt.ylabel('Minimum validation error')
plt.xticks(x1, datasets)
plt.legend()
plt.autoscale()
# plt.show()
plt.savefig(results_home + 'figures/ranking_validation_error.jpg')
plt.close()


