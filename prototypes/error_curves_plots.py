'''
This file is for showing the error curves of algorithms w.r.t 
each dataset on test error and validation error.
'''

import pickle
import os
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
from prototypes.data_analytic_pipeline import image_classification_pipeline


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, label=None):
	ax = ax if ax is not None else plt.gca()
	ymax = 0
	ymin = 0
	if color is None:
		color = ax._get_lines.color_cycle.next()
	if np.isscalar(yerr) or len(yerr) == len(y):
		ymin = y - yerr
		ymax = y + yerr
	elif len(yerr) == 2:
		ymin, ymax = yerr
	ax.plot(x, y, color=color, label=label)
	ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)



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

# TPE
start = 1
stop = 6
type1 = 'TPE'
TPE_val_error = []
# Output is a list of lists representing the errors w.r.t their times.
for data_name in datasets:
	val = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/TPE/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl', 'rb'), encoding='latin1')

		res = obj[1].results
		r0 = res[0]['eval_time']
		v = []
		for r in res:
			v.append((r['loss'], r['eval_time'] - r0))
		val.append(v)
	TPE_val_error.append(val)

# Random
start = 1
stop = 6
type1 = 'random_MCMC'
random_val_error = []
# Output is a list of lists representing the errors w.r.t their times.
for data_name in datasets:
	val = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/random/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl', 'rb'), encoding='latin1')

		times = np.asarray(obj.times)
		times -= times[0]
		times = times.tolist()
		v = []
		for i in range(len(times)):
			v.append((obj.error_curve[i], times[i]))
		val.append(v)
	random_val_error.append(val)

# Stochastic search
start = 1
stop = 6
type1 = 'RL_MCMC'
SS_val_error = []
# Output is a list of lists representing the errors w.r.t their times.
for data_name in datasets:
	val = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/SS/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full_multivariate.pkl', 'rb'), encoding='latin1')

		times = np.asarray(obj.times)
		times -= times[0]
		times = times.tolist()
		v = []
		for i in range(len(times)):
			v.append((obj.error_curve[i], times[i]))
		val.append(v)
	SS_val_error.append(val)


# # SMAC
# start = 1
# stop = 6
# type1 = 'bayesian_MCMC'
# SMAC_val_error = []
# # Output is a list of lists representing the errors w.r.t their times.
# for data_name in datasets:
# 	val = []
# 	for run in range(start, stop):
# 		obj = pickle.load(open(results_home + 'intermediate/SMAC/' + type1 + '_' + data_name + '_run_' +
# 							   str(run) + '_full.pkl', 'rb'), encoding='latin1')
#
# 		times = np.asarray(obj.times)
# 		times -= times[0]
# 		times = times.tolist()
# 		v = []
# 		for i in range(len(times)):
# 			v.append((obj.error_curves[i], times[i]))
# 		val.append(v)
# 	SMAC_val_error.append(val)


for i in range(len(datasets)):
	# Random
	max_time = 0
	min_time = np.inf
	for j in range(len(random_val_error[i])):
		z = random_val_error[i][j]
		if z[0][1] < min_time:
			min_time = z[0][1]
		if z[-1][1] > max_time:
			max_time = z[-1][1]

	times = np.linspace(min_time, max_time, 100)
	random_errors = []
	for j in range(len(random_val_error[i])):
		z = random_val_error[i][j]
		r = []
		for u in range(len(times)):
			v = 0
			for v in range(len(z)):
				if z[v][1] > times[u]:
					break
			f = 0
			if v == 0 or v == len(z) - 1:
				f = v
			else:
				f = v - 1
			r.append(z[f][0])
		r = np.expand_dims(np.asarray(r), 0)
		if j == 0:
			random_errors = r
		else:
			random_errors = np.vstack((random_errors, r))

	# TPE
	max_time = 0
	min_time = np.inf
	for j in range(len(TPE_val_error[i])):
		z = TPE_val_error[i][j]
		if z[0][1] < min_time:
			min_time = z[0][1]
		if z[-1][1] > max_time:
			max_time = z[-1][1]

	times = np.linspace(min_time, max_time, 100)
	TPE_errors = []
	for j in range(len(TPE_val_error[i])):
		z = TPE_val_error[i][j]
		r = []
		for u in range(len(times)):
			v = 0
			for v in range(len(z)):
				if z[v][1] > times[u]:
					break
			f = 0
			if v == 0 or v == len(z) - 1:
				f = v
			else:
				f = v - 1
			r.append(z[f][0])
		r = np.expand_dims(np.asarray(r), 0)
		if j == 0:
			TPE_errors = r
		else:
			TPE_errors = np.vstack((TPE_errors, r))

	# # SMAC
	# max_time = 0
	# min_time = np.inf
	# for j in range(len(SMAC_val_error[i])):
	# 	z = SMAC_val_error[i][j]
	# 	if z[0][1] < min_time:
	# 		min_time = z[0][1]
	# 	if z[-1][1] > max_time:
	# 		max_time = z[-1][1]
	#
	# times = np.linspace(min_time, max_time, 100)
	# SMAC_errors = []
	# for j in range(len(SMAC_val_error[i])):
	# 	z = SMAC_val_error[i][j]
	# 	r = []
	# 	for u in range(len(times)):
	# 		v = 0
	# 		for v in range(len(z)):
	# 			if z[v][1] > times[u]:
	# 				break
	# 		f = 0
	# 		if v == 0 or v == len(z) - 1:
	# 			f = v
	# 		else:
	# 			f = v - 1
	# 		r.append(z[f][0])
	# 	r = np.expand_dims(np.asarray(r), 0)
	# 	if j == 0:
	# 		SMAC_errors = r
	# 	else:
	# 		SMAC_errors = np.vstack((SMAC_errors, r))

	# Stochastic search
	max_time = 0
	min_time = np.inf
	for j in range(len(SS_val_error[i])):
		z = SS_val_error[i][j]
		if z[0][1] < min_time:
			min_time = z[0][1]
		if z[-1][1] > max_time:
			max_time = z[-1][1]

	times = np.linspace(min_time, max_time, 100)
	SS_errors = []
	for j in range(len(SS_val_error[i])):
		z = SS_val_error[i][j]
		r = []
		for u in range(len(times)):
			v = 0
			for v in range(len(z)):
				if z[v][1] > times[u]:
					break
			f = 0
			if v == 0 or v == len(z) - 1:
				f = v
			else:
				f = v - 1
			r.append(z[f][0])
		r = np.expand_dims(np.asarray(r), 0)
		if j == 0:
			SS_errors = r
		else:
			SS_errors = np.vstack((SS_errors, r))

	ax = plt.subplot(111)
	errorfill(times, np.mean(random_errors, 0), np.std(random_errors, 0), color='r', label='random', ax=ax)
	# errorfill(times, np.mean(TPE_errors, 0), np.std(TPE_errors, 0), color='g', label='TPE', ax=ax)
	errorfill(times, np.mean(SS_errors, 0), np.std(SS_errors, 0), color='b', label='Stochastic search', ax=ax)
	# errorfill(times, np.mean(SMAC_errors, 0), np.std(SMAC_errors, 0), color='y', label='SMAC', ax=ax)

	plt.title('Validation error curves')
	plt.xlabel('Time (s)')
	plt.ylabel('Validation error')
	plt.legend()
	plt.savefig(results_home + 'figures/validation_curve_' + datasets[i] + '.jpg')
	plt.close()
