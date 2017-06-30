from sklearn import tree, metrics
from os import system

class clfTree(object):
	"""Wrapper class for sklearn.tree"""

	def __init__(self, featuresNames, targetName):
		"""Creates a classifier with specified parameters in tune"""
		self.clf = tree.DecisionTreeClassifier(splitter = 'best')
		self.features = featuresNames
		self.target = targetName

	def setTreeParams(self, params):
		#for k in params.keys():
		#	setattr(self.clf, k, params[k])
		self.clf.max_depth = int(params[0])
		self.clf.min_samples_split = int(params[1])
		self.clf.min_samples_leaf = int(params[2])
		self.clf.min_impurity_split = params[3]

	def fit(self, trainData):
		"""Fits classifier with trainData"""
		X = trainData.loc[:, self.features]
		Y = trainData.loc[:, self.target]
		self.clf.fit(X, Y)

	def draw(self):
		"""Draws a tree in .png in current folder"""
		tree.export_graphviz(self.clf, out_file = "tree.dot", feature_names = featureNames)
		system('dot -Tpng tree.dot -o tree.png')
		system('rm tree.dot')

	def predict(self, testData, crossValidate = False):
		"""Makes prediction of self.target values based on testData"""
		X = testData.loc[:, self.features]
		prediction = self.clf.predict(X)
		score = None

		if crossValidate:
			trueY = testData.loc[:, self.target]
			score = metrics.accuracy_score(prediction, trueY)

		return({'Y' : prediction, 'score' : score})
