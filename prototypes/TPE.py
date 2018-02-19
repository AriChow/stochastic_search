import pickle
import time
from prototypes.data_analytic_pipeline import image_classification_pipeline
from hyperopt import tpe, fmin, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

def objective(g):
	return{
		'loss': g.get_error(),
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        # -- attachments are handled differently
        'attachments':
            {'time_module': pickle.dumps(time.time)}
	}


@scope.define
def pipeline(fe, dr, la, data_name, data_loc):
	hyper = {}
	if fe['type'] == 'haralick':
		hyper['haralick_distance'] = fe['haralick_distance']
	elif dr['type'] == 'PCA':
		hyper['pca_whiten'] = dr['pca_whiten']
	elif dr['type'] == 'ISOMAP':
		hyper['n_neighbors'] = dr['n_neighbors']
		hyper['n_components'] = dr['n_components']
	if la['type'] == 'RF':
		hyper['n_estimators'] = dr['n_estimators']
		hyper['max_features'] = dr['max_features']
	elif la['type'] == 'SVM':
		hyper['svm_C'] = dr['svm_C']
		hyper['svm_gamma'] = dr['svm_gamma']

	path = [fe['type'], dr['type'], la['type']]
	g = image_classification_pipeline(hyper, ml_type='validation', data_name=data_name,
												  data_loc=data_loc, type1='TPE', fe=path[0], dr=path[1], la=path[2],
												  val_splits=3, test_size=0.2)
	g.run()
	return g


home = os.path.expanduser('~')
dataset = sys.argv[1]
place = sys.argv[2]  # Documents/research for beeblebrox; barn for CCNI
data_home = home + '/' + place + '/EP_project/data/'
results_home = home + '/' + place + '/EP_project/results/'

# Empty features directory
start = int(sys.argv[3])
end = int(sys.argv[4])
import glob
files = glob.glob(data_home + 'features/random/*.npz')
for f in files:
	if os.path.exists(f):
		os.remove(f)
fe = hp.choice('feature_extraction', [
	{
		'type': 'haralick',

		'haralick_distance': hp.choice('haralick_distance', [1, 2, 3])
	},
	{
		'type': 'VGG'
	},
	{
		'type': 'inception'
	}
])

fe = hp.choice('dimensionality_reduction', [
	{
		'type': 'PCA',

		'pca_whiten': hp.choice('pca_whiten', [True, False])
	},
	{
		'type': 'ISOMAP',
		'n_neighbors': hp.choice('n_neighbors', [3, 4, 5, 6, 7])
		'n_components': hp.choice('n_components', [2, 3, 4])
	}
])

la = hp.choice('learning_algorithm',[
	{
		'type': 'RF',
		'n_estimators': hp.choice('n_estimators', range(8, 300))
		'max_features': hp.uniform('max_features', 0.3, 0.8)
	},
	{
		'type': 'SVM',
		'svm_gamma': hp.uniform('svm_gamma', 0.01, 8)
		'svm_C': hp.uniform('svm_gamma', 0.1, 100) 
	}
])

trials = Trials()
best = fmin(objective,
    space=pipeline(fe, dr, la, dataset, data_home),
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)

print best

