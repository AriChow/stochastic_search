import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mahotas.features import haralick
import cv2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import glob

class image_classification_pipeline(object):
	def __init__(self, hyper, ml_type=None, data_name=None, data_loc=None, type1=None, val_splits=None, test_size=None, fe=None, dr=None, la=None):
		self.feature_extraction = fe
		self.dimensionality_reduction = dr
		self.learning_algorithm = la
		self.ml_type = ml_type
		self.data_name = data_name
		self.data_location = data_loc
		self.val_splits = val_splits
		self.test_size = test_size
		self.f1_score = 0
		self.accuracy = 0
		self.result = None
		self.type1 = type1
		# r = glob.glob(self.data_location + 'features/' + self.type1 + '/*.npz')
		# for i in range(len(r)):
		# 	if self.feature_extraction in r[i] and self.feature_extraction == 'haralick':
		# 		os.remove(r[i])
		for key in hyper.keys():
			self.__setattr__(key, hyper[key])
		self.kwargs = hyper

	def get_error(self):
		return self.result

	def get_accuracy(self):
		return self.accuracy

	def get_f1_score(self):
		return self.f1_score

	def run(self):
		# Load the data
		data_home = self.data_location + 'datasets/' + self.data_name + '/'
		l1 = os.listdir(data_home)
		y = []
		names = []
		res = 0
		for z in range(len(l1)):
			if l1[z][0] == '.':
				continue
			l = os.listdir(data_home + l1[z] + '/')
			y += [z] * len(l)
			for i in range(len(l)):
				names.append(data_home + l1[z] + '/' + l[i])
		if self.ml_type == 'validation':
			# Train val split
			X = np.empty((len(y), 1))
			indices = np.arange(len(y))
			X1, _, y1, y_val, id1, _ = train_test_split(X, y, indices, test_size=self.test_size, random_state=42, shuffle=True)
			s = []
			# val_splits = 5
			kf = StratifiedKFold(n_splits=self.val_splits, random_state=42, shuffle=True)
			names1 = []
			for i in range(len(id1)):
				names1.append((names[id1[i]]))
			f1 = []
			acc = []
			res1 = []
			for idx1, idx2 in kf.split(X1, y1):
				X_train = []
				y_train = []
				ids1 = []
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
				a, r, f, _ = self.run_pipeline(names, y_train, y_val, ids1, ids2)
				res1.append(r)
				f1.append(f)
				acc.append(a)
			res = np.mean(res1)
			self.f1_score = np.mean(f1)
			self.accuracy = np.mean(acc)
		elif self.ml_type == 'testing':
			# Train val split
			X = np.empty((len(y), 1))
			indices = np.arange(len(y))
			_, _, y_train, y_val, idx1, idx2 = train_test_split(X, y, indices, test_size=self.test_size, random_state=42,
																shuffle=True)
			a, r, f, _ = self.run_pipeline(names, y_train, y_val, idx1, idx2)
			res = r
			self.f1_score = f
			self.accuracy = a
		# import glob
		# files = glob.glob(self.data_location + 'features/*.npz')
		# for f in files:
		# 	os.remove(f)
		self.result = res

	def run_pipeline(self, names, y_train, y_val, idx1, idx2):
		# Feature extraction
		f_train = []
		# f_test = []
		f_val = []
		if self.feature_extraction == "haralick":
			f_val = self.haralick_all_features(names, idx2, self.haralick_distance)
			f_train = self.haralick_all_features(names, idx1, self.haralick_distance)
		elif self.feature_extraction == "VGG":
			f_val = self.VGG_all_features(names, idx2)
			f_train = self.VGG_all_features(names, idx1)
		elif self.feature_extraction == "inception":
			f_val = self.inception_all_features(names, idx2)
			f_train = self.inception_all_features(names, idx1)
		if self.feature_extraction == "naive_feature_extraction":
			f_val = self.naive_all_features(names, idx2)
			f_train = self.naive_all_features(names, idx1)

		# Dimensionality reduction
		if self.dimensionality_reduction == "PCA":
			dr = self.principal_components(f_train, self.pca_whiten)
			f_train = dr.transform(f_train)
			f_val = dr.transform(f_val)

		elif self.dimensionality_reduction == "ISOMAP":
			dr = self.isomap(f_train, self.n_neighbors, self.n_components)
			f_train = dr.transform(f_train)
			f_val = dr.transform(f_val)

		elif self.dimensionality_reduction == "naive_dimensionality_reduction":
			dr = self.naive_transform(f_train)
			f_train = f_train
			f_val = f_val

		# Pre-processing
		normalizer = StandardScaler().fit(f_train)
		f_train = normalizer.transform(f_train)
		f_val = normalizer.transform(f_val)

		# Learning algorithms
		clf = []
		if self.learning_algorithm == "RF":
			clf = self.random_forests(f_train, y_train, int(self.n_estimators), self.max_features)
		elif self.learning_algorithm == "SVM":
			clf = self.support_vector_machines(f_train, y_train, self.svm_C, self.svm_gamma)
		elif self.learning_algorithm == "naive_learning_algorithm":
			clf = self.knn(f_train, y_train)
		# Metrics
		y_pred = clf.predict(f_val)
		p_pred = clf.predict_proba(f_val)
		conf = metrics.confusion_matrix(y_val, y_pred)
		err = metrics.log_loss(y_val, p_pred)
		f1 = metrics.f1_score(y_val, y_pred, average='weighted')
		acc = metrics.accuracy_score(y_val, y_pred)
		return acc, err, f1, conf

	def naive_all_features(self, names, idx):
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
		return f[idx, :]


	def haralick_all_features(self, names, idx, distance=1):
		# if os.path.exists(self.data_location + 'features/' + self.type1 + '/haralick_' + self.data_name + '.npz'):
		# 	f = np.load(open(self.data_location + 'features/' + self.type1 + '/haralick_' + self.data_name + '.npz', 'rb'))
		# 	return f.f.arr_0[idx, :]
		# else:
		f = []
		for i in range(len(names)):
			I = cv2.imread(names[i])
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
		# np.savez(open(self.data_location + 'features/' + self.type1 + '/haralick_' + self.data_name + '.npz', 'wb'), f)
		return f[idx, :]

	def CNN_all_features(self, names, cnn):
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


	def VGG_all_features(self, names, X):
		if os.path.exists(self.data_location + 'features/' + self.type1 + '/VGG_' + self.data_name + '.npz'):
			f = np.load(open(self.data_location + 'features/' + self.type1 + '/VGG_' + self.data_name + '.npz', 'rb'))
			return f.f.arr_0[X, :]
		else:
			f = self.CNN_all_features(names, 'VGG')
			# np.savez(open(self.data_location + 'features/' + self.type1 + '/VGG_' + self.data_name + '.npz', 'wb'), f)
			return f[X, :]

	def inception_all_features(self, names, X):
		if os.path.exists(self.data_location + 'features/' + self.type1 + '/inception_' + self.data_name + '.npz'):
			f = np.load(open(self.data_location + 'features/' + self.type1 + '/inception_' + self.data_name + '.npz', 'rb'))
			return f.f.arr_0[X, :]
		else:
			f = self.CNN_all_features(names, 'inception')
			# np.savez(open(self.data_location + 'features/' + self.type1 + '/inception_' + self.data_name + '.npz', 'wb'), f)
			return f[X, :]


	def naive_transform(self, X):
		return X

	def principal_components(self, X, whiten=False):
		pca = PCA(whiten=whiten)
		maxvar = 0.95
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

	def knn(self, X, y, neighbors=1):
		clf = KNeighborsClassifier(n_neighbors=neighbors)
		clf.fit(X, y)
		return clf

	def isomap(self, X, n_neighbors, n_components):
		iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
		iso.fit(X)
		return iso

	def random_forests(self, X, y, n_estimators, max_features):
		clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
									 class_weight='balanced')
		clf.fit(X, y)
		return clf

	def support_vector_machines(self, X, y, C, gamma):
		clf = svm.SVC(C=C, gamma=gamma, class_weight='balanced', probability=True)
		clf.fit(X, y)
		return clf

