from sklearn import tree, metrics
from os import system
from scipy import optimize

import matplotlib.pyplot as plt

class clfTree(object):
	"""Wrapper class for sklearn.tree"""

	def __init__(self, featuresNames, targetName):
		"""Creates a classifier with specified parameters in tune"""
		self.clf = tree.DecisionTreeClassifier(splitter = 'best')
		self.features = featuresNames
		self.target = targetName

	"""def setTreeParams(self, params):
					for k in params.keys():
						setattr(self.clf, k, params[k])"""

	def setImpurity(self, value):
		self.clf.min_impurity_split = value

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

	def fitOptimal(self, trainData, testData, domain = (0.001, 0.5)):
		"""Fits a tree from train data and chooses optimal tree parameters according to testData"""
		def lossFunction(imp):
			self.setImpurity(imp)
			self.fit(trainData)
			return(1. - self.predict(testData, crossValidate = True)['score'])

		solution = optimize.minimize_scalar(fun = lossFunction,
											method = 'bounded',
											bounds = domain,)
		if solution.success:
			optimum = (round(solution.x, 6), round(solution.fun, 6),)
		else:
			optimum = None
		
		return(optimum)