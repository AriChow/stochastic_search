import pickle
import os
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt


home = os.path.expanduser('~')
datasets = ['breast', 'matsc_dataset1', 'matsc_dataset2', 'brain', 'bone']
colors = ['b', 'k', 'y', 'r', 'g']
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
pipeline = {}
steps = ['feature_extraction', 'dimensionality_reduction', 'learning_algorithm']
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']

# Random1
start = 1
stop = 6
type1 = 'random_MCMC'
total_pipelines = []
for id, data_name in enumerate(datasets):
	best_pipelines = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl', 'rb'), encoding='latin1')
		min_err = 100000000
		pipelines = obj.pipelines
		p = []
		for i in pipelines:
			p.append((i, i.get_error()))
		sp = sorted(p, key=lambda p1: p1[1])
		best_pipelines = list(set(best_pipelines) | set(sp[:5]))
	best_pipelines = sorted(best_pipelines, key=lambda p1: p[1])
	algs = []
	for i, bp in enumerate(best_pipelines):
		algs.append((bp[0].feature_extraction.decode('latin1'), bp[0].dimensionality_reduction.decode('latin1') \
						, bp[0].learning_algorithm.decode('latin1')))
	algs = list(set(algs))
	bps = []
	for i, alg in enumerate(algs):
		for j, bp in enumerate(best_pipelines):
			if alg == (bp[0].feature_extraction.decode('latin1'), bp[0].dimensionality_reduction.decode('latin1') \
						, bp[0].learning_algorithm.decode('latin1')):
				bps.append(bp)
				break
	total_pipelines.append(bps)

fig = plt.figure()
ax = fig.add_subplot(111)
x = range(1, 6)
for i, data_name in enumerate(datasets):
	bps = total_pipelines[i]
	for bp in bps:
		ax.plot([i+1], [bp[1]], colors[i]+'x')
		ax.text(i+1, bp[1], '(' + bp[0].feature_extraction.decode('latin1') + ',' + bp[0].dimensionality_reduction.decode('latin1') + ',' + bp[0].learning_algorithm.decode('latin1') + ')', fontsize=10)
# plt.legend()
plt.title('Best paths')
plt.xlabel('Datasets')
plt.ylabel('Errors')
plt.xticks(x, datasets)
plt.savefig(results_home + 'figures/best_paths_datasets.jpg')
plt.close()

