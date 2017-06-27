import numpy
import matplotlib.pyplot as plt
from pandas import read_csv, Series, DataFrame, isnull
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

def fillField(col, opt = 'mode'):
	"""Replaces NaN values with appropriate (mode)"""
	if not col[isnull(col)].empty:
		filledFields = col[~isnull(col)]
		if opt == 'mode':
			replace = stats.mode(filledFields)[0][0]
		else:
			replace = numpy.mean(col)
		col[isnull(col)] = replace
	return(col)

def parseColumn(col, encod = {}, fillMode = "mode"):
	newCol = fillField(col, opt = fillMode)
	if encod:
		newCol = [encod[key] for key in newCol]
	return(newCol)

def prepareDataSet(path):
	dataFrame = loadData(path)
	
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
	sexEncoding = {'female' : 0, 'male' : 1}
	citiesEncoding = {'Q' : 0, 'C' : 1, 'S' : 2}
	
	titles = Series(grabAttribute('title', dataFrame.loc[:,"Name"]))
	dataFrame["Rank"] = parseColumn(titles, titlesEncoding)
	dataFrame["Sex"] = parseColumn(dataFrame["Sex"], sexEncoding)
	dataFrame["Embarked"] = parseColumn(dataFrame["Embarked"], citiesEncoding)
	dataFrame["Age"] = parseColumn(dataFrame["Age"])
	dataFrame["Fare"] = parseColumn(dataFrame["Fare"],  fillMode = "mean")
	
	# dropping irrelevant columns
	dataFrame.drop(["Name", "Ticket", "Cabin"], inplace = True, axis = 1)
	
	# reducing # of distinct values for some parameters
	dataFrame.loc[:,"Fare"] = dataFrame.loc[:,"Fare"].astype(int)
	dataFrame.loc[:,"Age"] = dataFrame.loc[:,"Age"].astype(int)
	return(dataFrame)

def Sample(dataframe, size):
	"""Splits dataframe randomly on 2 pieces with one sized <size>"""
	return()


def createTree(dataframe, features, target, tune):
	X = []
	iterFeatures = dataframe.loc[:,features].iterrows()
	for row in iterFeatures:
		X.append(list(row[1]))
	Y = list(dataframe.loc[:,target])

	clf = tree.DecisionTreeClassifier(splitter = 'best', **tune)
	X = dataframe.loc[:,features]
	Y = list(dataframe.loc[:,target])
	clf.fit(X, Y)
	return(clf)

def drawTree(classifier, featureNames):
	tree.export_graphviz(classifier, out_file = "tree.dot", feature_names = featureNames)
	system('dot -Tpng tree.dot -o tree.png')

if __name__ == '__main__':
	dfTrain = prepareDataSet("train.csv")
	features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Rank", "Sex", "Embarked"]

	treeParams = {"max_depth" : 10, "min_samples_leaf" : 10, "min_impurity_split" : 0.175}

	clfTree = createTree(dfTrain, features, "Survived", treeParams)
	drawTree(clfTree, features)
	
	dfTest = prepareDataSet("test.csv")
	result = clfTree.predict(dfTest.ix[:, dfTest.columns != "PassengerId"])

	predictions = dfTest["PassengerId"].to_frame()
	predictions["Survived"] = Series(result)
	submission = predictions.to_csv('submission.csv', header = True, index = False)