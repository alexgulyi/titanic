import numpy
import matplotlib.pyplot as plt
from pandas import read_csv, Series
from re import search, match
from scipy import stats

def LoadData(path):
	"""Loads a file into dataframe"""
	try:
		return(read_csv(path))
	except OSError:
		return(None)

def GrabAttribute(attr, names):
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

def MapFamilies(names):
	"""Takes names array and returns dict with <surname>:<# of family members>"""
	familyDict = {}
	families = numpy.array(GrabAttribute('surname', names))
	familyUnique = set(families)
	for f in familyUnique:
		familyDict[f] = len(numpy.where(families == f)[0])
	return(familyDict)

def FillAges(ages):
	"""Replaces NaN values with appropriate (mode)"""
	filledAges = dataFrame.loc[~numpy.isnan(dataFrame.loc[:,"Age"])].loc[:,"Age"]
	mode = stats.mode(filledAges)[0][0]
	ages[numpy.isnan(ages)] = mode
	return(ages)


if __name__ == '__main__':
	dataFrame = LoadData('train.csv')
	titles = GrabAttribute('title', dataFrame.loc[:,"Name"])
	newAges = FillAges(dataFrame.loc[:,"Age"])

	dataFrame["Age"] = Series(newAges)
	dataFrame["Titles"] = Series(titles)

	print(dataFrame.head())