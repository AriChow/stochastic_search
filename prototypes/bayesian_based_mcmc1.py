from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np
import copy
import pickle
import time
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

class bayesian_MCMC():
	def __init__(self, data_name, data_loc, results_loc, run, pipeline):
		self.pipeline = pipeline
		self.paths = []
		self.best_pipelines = []
		self.potential = []
		self.times = []
		self.error_curves = []
		self.incumbents = []
		self.all_incumbents = []
		self.data_name = data_name
		self.data_loc = data_loc
		self.run = run
		self.results_loc = results_loc

	def populate_paths(self):
		pipeline = self.pipeline
		paths = []
		for i in pipeline['feature_extraction']:
			path = [i]
			for j in pipeline['dimensionality_reduction']:
				path1 = copy.deepcopy(path)
				path1.append(j)
				for k in pipeline['learning_algorithm']:
					path2 = copy.deepcopy(path1)
					path2.append(k)
					paths.append(path2)
		self.paths = paths

	def bayesianmcmc(self):
		from sklearn.decomposition import PCA
		from sklearn.manifold import Isomap
		from sklearn import svm
		from sklearn.ensemble import RandomForestClassifier
		from mahotas.features import haralick
		import os
		from sklearn.model_selection import StratifiedKFold, train_test_split
		from sklearn import metrics
		from sklearn.preprocessing import StandardScaler
		import cv2
		from sklearn.neighbors import KNeighborsClassifier

		def naive_all_features(names):
			f = []
			for i in range(len(names)):
				I = cv2.imread(names[i])
				l = I.shape
				f1 = []
				if I is None or I.size == 0 or np.sum(I[:]) == 0 or I.shape[0] == 0 or I.shape[1] == 0:
					if len(l) == 3:
						f1 = np.zeros((1, l[0] * l[1] * l[2]))
				else:
					f1 = I.flatten()
				f1 = np.expand_dims(f1, 0)
				if i == 0:
					f = f1
				else:
					f = np.vstack((f, f1))
			return f

		def haralick_all_features(X, distance=1):
			f = []
			for i in range(len(X)):
				I = cv2.imread(X[i])
				if I is None or I.size == 0 or np.sum(I[:]) == 0 or I.shape[0] == 0 or I.shape[1] == 0:
					h = np.zeros((1, 13))
				else:
					I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
					h = haralick(I, distance=distance, return_mean=True, ignore_zeros=False)
					h = np.expand_dims(h, 0)
				if i == 0:
					f = h
				else:
					f = np.vstack((f, h))
			return f

		def CNN_all_features(names, cnn):
			from keras.applications.vgg19 import VGG19
			from keras.applications.inception_v3 import InceptionV3
			from keras.applications.vgg19 import preprocess_input
			f = []
			if cnn == 'VGG':
				model = VGG19(weights='imagenet')
				dsize = (224, 224)
			else:
				model = InceptionV3(weights='imagenet')
				dsize = (299, 299)
			for i in range(len(names)):
				img = cv2.imread(names[i])
				img = cv2.resize(img, dsize=dsize)
				img = img.astype('float32')
				x = np.expand_dims(img, axis=0)
				x = preprocess_input(x)
				features = model.predict(x)
				if i == 0:
					f = features
				else:
					f = np.vstack((f, features))
			return f

		def VGG_all_features(names, X):
			home = os.path.expanduser('~')
			if os.path.exists(self.data_loc + 'features/bayesian1/VGG_' + self.data_name + '.npz'):
				f = np.load(open(self.data_loc + 'features/bayesian1/VGG_' + self.data_name + '.npz', 'rb'))
				return f.f.arr_0[X, :]
			else:
				f = CNN_all_features(names, 'VGG')
				np.savez(open(self.data_loc + 'features/bayesian1/VGG_' + self.data_name + '.npz', 'wb'), f)
				return f[X, :]

		def inception_all_features(names, X):
			home = os.path.expanduser('~')
			if os.path.exists(self.data_loc + 'features/bayesian1/inception_' + self.data_name + '.npz'):
				f = np.load(open(self.data_loc + 'features/bayesian1/inception_' + self.data_name + '.npz', 'rb'))
				return f.f.arr_0[X, :]
			else:
				f = CNN_all_features(names, 'inception')
				np.savez(open(self.data_loc + 'features/bayesian1/inception_' + self.data_name + '.npz', 'wb'), f)
				return f[X, :]

		def principal_components(X, whiten=True):
			pca = PCA(whiten=whiten)
			maxvar = 0.95
			X = np.asarray(X)
			if len(X.shape) == 1:
				X = X.reshape(-1, 1)
			data = X
			X1 = pca.fit(X)
			var = pca.explained_variance_ratio_
			s1 = 0
			for i in range(len(var)):
				s1 += var[i]
			s = 0
			for i in range(len(var)):
				s += var[i]
				if (s * 1.0 / s1) >= maxvar:
					break
			pca = PCA(n_components=i + 1)
			pca.fit(data)
			return pca

		def isomap(X, n_neighbors=5, n_components=2):
			iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
			X = np.asarray(X)
			if len(X.shape) == 1:
				X = X.reshape(-1, 1)
			iso.fit(X)
			return iso

		def random_forests(X, y, n_estimators, max_features):
			clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
										 class_weight='balanced')
			clf.fit(X, y)
			return clf

		def support_vector_machines(X, y, C, gamma):
			clf = svm.SVC(C=C, gamma=gamma, class_weight='balanced', probability=True)
			clf.fit(X, y)
			return clf

		def knn(X, y, neighbors=1):
			clf = KNeighborsClassifier(n_neighbors=neighbors)
			clf.fit(X, y)
			return clf

		def pipeline_from_cfg(cfg):
			cfg = {k: cfg[k] for k in cfg if cfg[k]}
			# Load the data
			data_home = self.data_loc + 'datasets/' + self.data_name + '/'
			l1 = os.listdir(data_home)
			y = []
			names = []
			cnt = 0
			for z in range(len(l1)):
				if l1[z][0] == '.':
					continue
				l = os.listdir(data_home + l1[z] + '/')
				y += [cnt] * len(l)
				cnt += 1
				for i in range(len(l)):
					names.append(data_home + l1[z] + '/' + l[i])
			# Train val split
			X = np.empty((len(y), 1))
			indices = np.arange(len(y))
			X1, _, y1, y_val, id1, _ = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)
			s = []
			val_splits = 3
			kf = StratifiedKFold(n_splits=val_splits, random_state=42, shuffle=True)
			names1 = []
			for i in range(len(id1)):
				names1.append((names[id1[i]]))
			f11 = []
			for idx1, idx2 in kf.split(X1, y1):
				# Feature extraction
				ids1 = []
				X_train = []
				y_train = []
				for i in idx1:
					X_train.append(names1[i])
					y_train.append(y1[i])
					ids1.append(id1[i])
				X_val = []
				y_val = []
				ids2 = []
				for i in idx2:
					X_val.append(names1[i])
					y_val.append(y1[i])
					ids2.append(id1[i])
				# Feature extraction
				f_train = []
				# f_test = []
				f_val = []
				if cfg['feature_extraction'] == "haralick":
					f_val = haralick_all_features(X_val, cfg["haralick_distance"])
					f_train = haralick_all_features(X_train, cfg["haralick_distance"])
				elif cfg['feature_extraction'] == "VGG":
					f_val = VGG_all_features(names, ids2)
					f_train = VGG_all_features(names, ids1)
				elif cfg['feature_extraction'] == "inception":
					f_val = inception_all_features(names, ids2)
					f_train = inception_all_features(names, ids1)
				elif cfg['feature_extraction'] == "naive_feature_extraction":
					f_val = naive_all_features(X_val)
					f_train = naive_all_features(X_train)

				# Dimensionality reduction
				if cfg['dimensionality_reduction'] == "PCA":
					cfg["pca_whiten"] = True if cfg["pca_whiten"] == "true" else False
					dr = principal_components(f_train, cfg["pca_whiten"])
					f_train = dr.transform(f_train)
					f_val = dr.transform(f_val)

				elif cfg['dimensionality_reduction'] == "ISOMAP":
					dr = isomap(f_train, cfg["n_neighbors"], cfg["n_components"])
					f_train = dr.transform(f_train)
					f_val = dr.transform(f_val)

				elif cfg['dimensionality_reduction'] == 'naive_dimensionality_reduction':
					f_train = f_train
					f_val = f_val

				# Pre-processing
				normalizer = StandardScaler().fit(f_train)
				f_train = normalizer.transform(f_train)
				f_val = normalizer.transform(f_val)

				# Learning algorithms
				if cfg['learning_algorithm'] == "RF":
					clf = random_forests(f_train, y_train, cfg["n_estimators"], cfg["max_features"])
				elif cfg['learning_algorithm'] == "SVM":
					clf = support_vector_machines(f_train, y_train, cfg["svm_C"], cfg["svm_gamma"])
				elif cfg['learning_algorithm'] == 'naive_learning_algorithm':
					clf = knn(f_train, y_train)
				p_pred = clf.predict_proba(f_val)
				f11.append(metrics.log_loss(y_val, p_pred))
				s.append(clf.score(f_val, y_val))
			return np.mean(f11)



		self.potential = []
		self.best_pipelines = []
		self.times = []
		self.error_curves = []
		cs = ConfigurationSpace()
		feature_extraction = CategoricalHyperparameter("feature_extraction", ["haralick", "VGG", "inception"],
													  default="haralick")
		cs.add_hyperparameter(feature_extraction)

		dimensionality_reduction = CategoricalHyperparameter("dimensionality_reduction", ["PCA", "ISOMAP"],
															 default="PCA")
		cs.add_hyperparameter(dimensionality_reduction)

		learning_algorithm = CategoricalHyperparameter("learning_algorithm", ["SVM", "RF"], default="RF")
		cs.add_hyperparameter(learning_algorithm)


		haralick_distance = UniformIntegerHyperparameter("haralick_distance", 1, 3, default=1)
		cs.add_hyperparameter(haralick_distance)
		cond1 = InCondition(child=haralick_distance, parent=feature_extraction, values=["haralick"])
		cs.add_condition(cond1)

		pca_whiten = CategoricalHyperparameter("pca_whiten", ["true", "false"], default="true")
		cs.add_hyperparameter(pca_whiten)
		cs.add_condition(InCondition(child=pca_whiten, parent=dimensionality_reduction, values=["PCA"]))

		n_neighbors = UniformIntegerHyperparameter("n_neighbors", 3, 7, default=5)
		n_components = UniformIntegerHyperparameter("n_components", 2, 4, default=2)
		cs.add_hyperparameters([n_neighbors, n_components])
		cs.add_condition(InCondition(child=n_components, parent=dimensionality_reduction, values=["ISOMAP"]))
		cs.add_condition(InCondition(child=n_neighbors, parent=dimensionality_reduction, values=["ISOMAP"]))

		svm_C = UniformFloatHyperparameter("svm_C", 0.1, 100.0, default=1.0)
		cs.add_hyperparameter(svm_C)
		svm_gamma = UniformFloatHyperparameter("svm_gamma", 0.01, 8, default=1)
		cs.add_hyperparameter(svm_gamma)
		cond1 = InCondition(child=svm_C, parent=learning_algorithm, values=["SVM"])
		cond2 = InCondition(child=svm_gamma, parent=learning_algorithm, values=["SVM"])
		cs.add_conditions([cond1, cond2])

		n_estimators = UniformIntegerHyperparameter("n_estimators", 8, 300, default=10)
		max_features = UniformFloatHyperparameter("max_features", 0.3, 0.8, default=0.5)
		cs.add_hyperparameters([max_features, n_estimators])
		cond1 = InCondition(child=n_estimators, parent=learning_algorithm, values=["RF"])
		cond2 = InCondition(child=max_features, parent=learning_algorithm, values=["RF"])
		cs.add_conditions([cond1, cond2])

		scenario = Scenario({"run_obj": "quality",
							 "cutoff_time": 100000,
							 "runcount_limit": 10000 * 10,
							 "cs": cs,
							 "maxR": 100000,
							 "wallclock_limit" : 1000000,
							 "deterministic": "true"})
		smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=pipeline_from_cfg)
		incumbent, incs, incumbents, incumbents1, times = smac.optimize()
		inc_value = pipeline_from_cfg(incumbent)
		self.best_pipelines.append(incumbent)
		self.potential.append(inc_value)
		self.incumbents = incumbents
		self.all_incumbents = incumbents1
		self.error_curves.append(incs)
		self.times = times
		pickle.dump(self, open(self.results_loc + 'intermediate/SMAC/SMAC_' + self.data_name + '_run_' + str(self.run) + '_full.pkl', 'wb'))
