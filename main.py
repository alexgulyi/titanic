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

def fillAges(ages):
	"""Replaces NaN values with appropriate (mode)"""
	filledAges = ages[~numpy.isnan(ages)]
	#filledAges = dataFrame.loc[~numpy.isnan(dataFrame.loc[:,"Age"])].loc[:,"Age"]
	mode = stats.mode(filledAges)[0][0]
	ages[numpy.isnan(ages)] = mode
	return(ages)

def prepareDataSet(path):
	dataFrame = loadData(path)
	titles = grabAttribute('title', dataFrame.loc[:,"Name"])
	newAges = fillAges(dataFrame.loc[:,"Age"])
	dataFrame["Age"] = Series(newAges)
	dataFrame["Title"] = Series(titles)
	# dropping columns that don't have enough info
	dataFrame.drop(["Sex", "PassengerId", "Name", "Ticket"], inplace = True, axis = 1)
	return(dataFrame)

def createTree(dataframe, features, target):
	clf = tree.DecisionTreeClassifier()
	X = dataframe.loc[:,features]
	Y = list(dataframe.loc[:,target])

if __name__ == '__main__':
	df = prepareDataSet("train.csv")
	print(df.head())
	# features preparation
	X = []
	iterFeatures = df.loc[:,["Pclass", "Age", "SibSp", "Parch", "Fare", "Title"]].iterrows()
	for row in iterFeatures:
		X.append(list(row[1]))
	Y = list(df.loc[:,"Survived"])