import numpy as np
import os
from prototypes.bayesian_based_mcmc1 import bayesian_MCMC
import sys

# if __name__==' __main__':
home = os.path.expanduser('~')
dataset = sys.argv[1]
place = sys.argv[2]  # Documents/research for beeblebrox; barn for CCNI
data_home = home + '/' + place + '/EP_project/data/'
results_home = home + '/' + place + '/stochastic_search/results/'
# start = int(sys.argv[3])
# end = int(sys.argv[4])
# Empty features directory
import glob
files = glob.glob(data_home + 'features/bayesian1/*.npz')
for f in files:
	if os.path.exists(f):
		os.remove(f)
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['haralick_distance'] = range(1, 4)
pipeline['pca_whiten'] = [True, False]
pipeline['n_neighbors'] = range(3, 8)
pipeline['n_components'] = range(2, 5)
pipeline['n_estimators'] = np.round(np.linspace(8, 300, 10))
pipeline['max_features'] = np.arange(0.3, 0.8, 0.1)
pipeline['svm_gamma'] = np.linspace(0.01, 8, 10)
pipeline['svm_C'] = np.linspace(0.1, 100, 10)

# CONTROL
for i in range(5):
	rm = bayesian_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, run=i+1, pipeline=pipeline)
	rm.populate_paths()
	rm.bayesianmcmc()
