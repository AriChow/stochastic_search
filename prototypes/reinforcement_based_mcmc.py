from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np
import copy
import pickle
import time
import os

class RL_MCMC():
	def __init__(self, data_name=None, data_loc=None, cutoff_time = None, results_loc=None, run=None, type1=None, pipeline=None,
				 path_resources=None, hyper_resources=None, iters=None):
		self.pipeline = pipeline
		self.paths = []
		self.pipelines = []
		self.times = []
		self.run = run
		self.path_resources = path_resources
		self.hyper_resources = hyper_resources
		self.potential = []
		self.best_pipelines = []
		self.data_name = data_name
		self.data_loc = data_loc
		self.iters = iters
		self.results_loc = results_loc
		self.type1 = type1
		self.error_curve = []
		self.path_probs = []
		self.hyper_info = {}
		self.last_t = 0
		self.last_cnt = 0
		self.cutoff_time = cutoff_time
		self.best_error = 100000

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

	def pick_hyper(self, pipeline, eps, t):
		p = eps * 1.0 / (t ** (1 / 4))
		r = np.random.uniform(0, 1, 1)
		pipeline = self.pipeline
		paths = self.paths
		path_probs = self.path_probs
		hyper_info = self.hyper_info
		hyper = {}
		if r[0] < p:
			# Pick hyper-parameters randomly (Exploration)
			r1 = np.random.choice(len(self.paths), 1)
			path = self.paths[r1[0]]
			if path[0] == 'haralick':
				r = np.random.choice(pipeline['haralick_distance'], 1)
				hyper['haralick_distance'] = r[0]
			if path[1] == 'PCA':
				r = np.random.choice(pipeline['pca_whiten'], 1)
				hyper['pca_whiten'] = r[0]
			elif path[1] == 'ISOMAP':
				r = np.random.choice(pipeline['n_neighbors'], 1)
				hyper['n_neighbors'] = r[0]
				r = np.random.choice(pipeline['n_components'], 1)
				hyper['n_components'] = r[0]
			if path[2] == 'RF':
				r = np.random.choice(pipeline['n_estimators'], 1)
				hyper['n_estimators'] = r[0]
				r = np.random.uniform(pipeline['max_features'], 1)
				hyper['max_features'] = r[0]
			elif path[2] == 'SVM':
				r = np.random.uniform(pipeline['svm_C'][0], pipeline['svm_C'][-1], 1)
				hyper['svm_C'] = r[0]
				r = np.random.uniform(pipeline['svm_gamma'][0], pipeline['svm_gamma'][-1], 1)
				hyper['svm_gamma'] = r[0]
		else:
			# Pick the algorithms and hyper-parameters using stochastic search (Exploitation)
			discrete = ['haralick_distance', 'pca_whiten', 'n_neighbors', 'n_estimators', 'n_components']
			r1 = np.random.choice(range(len(self.paths)), size=1, p=path_probs)
			path = paths[r1[0]]
			hyper1 = hyper_info[r1[0]]
			h1 = {}
			for h in hyper1:
				vals = []
				errs = []
				pipeline_values = self.pipeline[h]
				for v in hyper1[h]:
					vals.append(v)
					errs.append(hyper1[h][v])
				vals1 = copy.deepcopy(vals)
				if h not in discrete:
					bins = np.linspace(pipeline_values[0], pipeline_values[-1], 100)
					inds = np.digitize(vals, bins)
					errs1 = []
					for i in range(len(bins)):
						b = np.asarray(inds) == i
						c = np.asarray(errs)
						errs1.append(np.dot(b, c))
					vals = bins
					errs = errs1
				errs /= np.sum(errs)
				if len(vals) == 0:
					if h in discrete:
						r = np.random.choice(pipeline_values, 1)
						hyper[h] = r[0]
					else:
						r = np.random.uniform(pipeline_values[0], pipeline_values[-1], 1)
						hyper[h] = r[0]
				else:
					r1 = np.random.choice(range(len(vals)), size=1, p=errs)
					h1[h] = vals[r1[0]]
					if h in discrete:  # sample from 2 - neighborhood
						lenh = len(pipeline_values)
						sample_space = 5
						if lenh < 5:
							sample_space = lenh
						ind = pipeline_values.index(h1[h])
						possible_values = []
						for i1 in range(ind, -1, -1):
							if len(possible_values) > sample_space // 2:
								break
							possible_values.append(pipeline_values[i1])
						if ind < lenh - 1:
							for i1 in range(ind + 1, lenh):
								if len(possible_values) >= sample_space:
									break
								possible_values.append(pipeline_values[i1])
						r1 = np.random.choice(possible_values, 1)
						hyper[h] = r1[0]
					else:  # sample from region around parameter
						s = vals1
						std = 3.0 * np.std(s) / len(s)
						h_low = h1[h] - std
						h_high = h1[h] + std
						if h_low < self.pipeline[h][0] or len(s) == 1:
							h_low = self.pipeline[h][0]
						if h_high > self.pipeline[h][-1] or len(s) == 1:
							h_high = self.pipeline[h][-1]
						r1 = np.random.uniform(h_low, h_high, 1)
						# r1 = np.random.normal(h1[h], std, 1)
						hn = r1[0]
						# if hn < self.pipeline[h][0]:
						# 	hn = self.pipeline[h][0]
						# elif hn > self.pipeline[h][-1]:
						# 	hn = self.pipeline[h][-1]
						hyper[h] = hn
		return hyper, path

	def rlMcmc(self):
		eps = 1
		hypers = ['haralick_distance', 'pca_whiten', 'n_neighbors', 'n_components', 'n_estimators', 'max_features',
				  'svm_C', 'svm_gamma']
		discrete = ['haralick_distance', 'pca_whiten', 'n_neighbors', 'n_components', 'n_estimators']
		hyper_algs = ['haralick', 'PCA', 'ISOMAP', 'ISOMAP', 'RF', 'RF', 'SVM', 'SVM']

		if os.path.exists(self.results_loc + 'intermediate/SS/' + self.type1 + '_' + self.data_name + '_run_' + str(
						self.run) + '_full_last_object.pkl'):
			last_object = pickle.load(open(self.results_loc + 'intermediate/SS/' + self.type1 + '_' + self.data_name + '_run_' + str(
						self.run) + '_full_last_object.pkl', 'rb'))
			pipelines = last_object.pipelines
			best_pipelines = last_object.best_pipelines
			t = last_object.last_t
			times = last_object.times
			add_time = times[-1]
			best_error1 = last_object.best_error
			best_error = last_object.best_error
			self.path_probs = last_object.path_probs
			self.hyper_info = last_object.hyper_info
			cnt = last_object.last_cnt
		else:
			add_time = 0
			pipelines = []
			times = []
			best_pipelines = []
			t = 0
			best_error1 = 100000
			best_error = 100000
			cnt = 0
			self.path_probs = 1.0 / len(self.paths) * np.ones(len(self.paths))
			for i in range(len(self.paths)):
				self.hyper_info[i] = {}
				for j in range(len(hyper_algs)):
					if hyper_algs[j] in self.paths[i]:
						self.hyper_info[i][hypers[j]] = {}
		t0 = time.time()
		while (True):
			t += 1
			## MAIN STEP - CHOOSING THE NEXT HYPER-PARAMETERS AND ALGORITHMS
			hyper, path = self.pick_hyper(pipelines, eps, t)


			g = image_classification_pipeline(hyper, ml_type='validation', data_name=self.data_name,
											  data_loc=self.data_loc, type1='RL1', fe=path[0], dr=path[1], la=path[2],
											  val_splits=3, test_size=0.2)
			g.run()
			err = g.get_error()

			# Update path and hyper-parameter
			# err = np.random.random()
			pipelines.append((g, path))
			ind = self.paths.index(path)
			self.path_probs[ind] += 1.0 / err
			self.path_probs /= np.sum(self.path_probs)

			for h in self.hyper_info[ind]:
				if h in discrete:
					h1 = hyper[h]
				else:
					h1 = np.round(hyper[h], 2)
				if h1 in self.hyper_info[ind][h]:
					self.hyper_info[ind][h][h1] += 1.0 / err
				else:
					self.hyper_info[ind][h][h1] = 1.0 / err

			if err < best_error:
				best_error = err
			if best_error >= best_error1:
				cnt += 1
			else:
				cnt = 0
			if best_error1 > best_error:
				best_error1 = best_error
				best_pipelines.append((g, path))
			self.error_curve.append(best_error1)
			t1 = time.time()
			if cnt >= self.iters or t > 10000 or (t1-t0 + add_time) > self.cutoff_time:
				break
			times.append(t1 - t0 + add_time)
			self.pipelines = pipelines
			self.times = times
			self.best_pipelines = best_pipelines
			self.last_t = t
			self.last_cnt = cnt
			self.best_error1 = best_error1
			if t1 - t0 > 10000:
				pickle.dump(self, open(
					self.results_loc + 'intermediate/SS/' + self.type1 + '_' + self.data_name + '_run_' + str(
						self.run) + '_full_last_object.pkl',
					'wb'))

		pickle.dump(self, open(
			self.results_loc + 'intermediate/SS/' + self.type1 + '_' + self.data_name + '_run_' + str(
				self.run) + '_full.pkl',
			'wb'))
