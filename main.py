import numpy
import matplotlib.pyplot as plt
from pandas import read_csv, Series
from re import search, match
from scipy import stats
from os import system
from sklearn import tree

def loadData(path):
	"""Loads a file into dataframe"""
	try:
		return(read_csv(path))
	except OSError:
		return(None)

def grabAttribute(attr, names):
	"""Grab predefined attribute from name field value"""
	if attr == 'title':		
		pattern = "(?<=,\s)([^\.])*"
	elif attr == 'surname':
		pattern = "^([^,])*"
	else:
		return(None)
	
	result = []
	for name in names:
		try:
			match = search(pattern, name)
			result.append(match.group())
		except AttributeError:
			result.append("NONE")
			continue
	return(result)

def fillField(col, opt):
	"""Replaces NaN values with appropriate (mode)"""
	filledFields = col[~numpy.isnan(col)]
	if opt == 'mode':
		replace = stats.mode(filledFields)[0][0]
	else:
		replace = numpy.mean(col)
	col[numpy.isnan(col)] = replace
	return(col)

def prepareDataSet(path):
	dataFrame = loadData(path)
	titles = grabAttribute('title', dataFrame.loc[:,"Name"])
	
	# + it's possible to transform titles into integer rank:
	# 1 - no titles, 2 - job-related titles, 3 - nobility titles
	titlesEncoding = {'Col' : 2,
						'Miss' : 1,
						'Lady' : 3,
						'Rev' : 2,
						'the Countess' : 3,
						'Capt' : 2,
						'Sir' : 3,
						'Mme' : 3,
						'Dr' : 2,
						'Master' : 2,
						'Don' : 3,
						'Ms' : 1,
						'Mlle' : 1,
						'Major' : 2,
						'Jonkheer' : 3,
						'Mr' : 1,
						'Mrs' : 1,
						'Dona' : 3,}
	ranks = [titlesEncoding[t] for t in titles]

	#newAges = fillAges(dataFrame.loc[:,"Age"])
	newAges = fillField(dataFrame.loc[:,"Age"], opt = "mode")
	dataFrame["Age"] = Series(newAges)
	dataFrame["Rank"] = Series(ranks)

	newFares = fillField(dataFrame.loc[:,"Age"], opt = "mean")
	dataFrame["Fare"] = Series(newFares)
	
	# dropping columns that don't have enough info
	dataFrame.drop(["PassengerId", "Name", "Sex", "Ticket", "Cabin", "Embarked"], inplace = True, axis = 1)
	
	# reducing distinct values for some parameters
	dataFrame.loc[:,"Fare"] = dataFrame.loc[:,"Fare"].astype(int)
	dataFrame.loc[:,"Age"] = dataFrame.loc[:,"Age"].astype(int)
	return(dataFrame)

def createTree(dataframe, features, target):
	X = []
	iterFeatures = dataframe.loc[:,features].iterrows()
	for row in iterFeatures:
		X.append(list(row[1]))
	Y = list(dataframe.loc[:,target])

	clf = tree.DecisionTreeClassifier(splitter = 'random')
	X = dataframe.loc[:,features]
	Y = list(dataframe.loc[:,target])
	clf.fit(X, Y)
	return(clf)

def drawTree(classifier, featureNames):
	tree.export_graphviz(classifier, out_file = "tree.dot", feature_names = featureNames)
	system('dot -Tpng tree.dot -o tree.png')

if __name__ == '__main__':
	dfTrain = prepareDataSet("train.csv")
	print(dfTrain.head())
	features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Rank"]
	clfTree = createTree(dfTrain, features, "Survived")
	drawTree(clfTree, features)
	
	dfTest = prepareDataSet("test.csv")
	test = clfTree.predict(dfTest)
	print(clfTree.predict(dfTest))