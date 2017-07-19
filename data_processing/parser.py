from os import system
from pandas import DataFrame, Series, read_csv, isnull
from re import search, match
from scipy import stats
from numpy import mean

titlesEncoding = {'Col' : 0,
					'Miss' : 1,
					'Lady' : 2,
					'Rev' : 3,
					'the Countess' : 4,
					'Capt' : 5,
					'Sir' : 6,
					'Mme' : 7,
					'Dr' : 8,
					'Master' : 9,
					'Don' : 10,
					'Ms' : 11,
					'Mlle' : 12,
					'Major' : 13,
					'Jonkheer' : 14,
					'Mr' : 15,
					'Mrs' : 16,
					'Dona' : 17,}

"""titlesEncoding = {'Col' : 2,
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
						'Dona' : 3,}"""

sexEncoding = {'female' : 0,
				'male' : 1}
citiesEncoding = {'Q' : 0,
					'C' : 1,
					'S' : 2}

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
	elif attr == 'ticket':
		pattern = "\d+"
	else:
		return(None)
	
	result = []
	for name in names:
		try:
			match = search(pattern, name)
			result.append(match.group())
		except AttributeError:
			result.append(None)
			continue
	return(result)

def parseColumn(col, encod = {}, fillMode = "mode"):
	"""Fills NaN values in column and encodes strings into numbers if necessary"""
	
	def fillField(col, opt = 'mode'):
		"""Replaces NaN values with appropriate (mode, mean...)"""
		if not col[isnull(col)].empty:
			filledFields = col[~isnull(col)]
			if opt == 'mode':
				replace = stats.mode(filledFields)[0][0]
			# otherwise always mean (for yet)
			else:
				replace = mean(col)
			col.loc[isnull(col)] = replace
		return(col)

	newCol = fillField(col, opt = fillMode)
	if encod:
		newCol = [encod[key] for key in newCol]
	return(newCol)

def prepareDataSet(path):
	dataFrame = loadData(path)
	
	dataFrame["Rank"] = parseColumn(Series(grabAttribute('title', dataFrame.loc[:,"Name"])),
									titlesEncoding)
	dataFrame["Sex"] = parseColumn(dataFrame["Sex"], sexEncoding)
	dataFrame["Embarked"] = parseColumn(dataFrame["Embarked"], citiesEncoding)
	dataFrame["Age"] = parseColumn(dataFrame["Age"])
	dataFrame["Fare"] = parseColumn(dataFrame["Fare"],  fillMode = "mean")
	dataFrame["Ticket"] = parseColumn(Series(grabAttribute('ticket', dataFrame.loc[:, "Ticket"])))
	
	# dropping irrelevant columns
	dataFrame.drop(["Name", "Cabin"], inplace = True, axis = 1)
	
	# reducing # of distinct values for some parameters
	dataFrame.loc[:,"Fare"] = dataFrame.loc[:,"Fare"].astype(int)
	dataFrame.loc[:,"Age"] = dataFrame.loc[:,"Age"].astype(int)
	dataFrame.loc[:,"Ticket"] = dataFrame.loc[:,"Ticket"].astype(int)

	return(dataFrame)

def packResult(ids, results):
	dataFrame = DataFrame.from_records(zip(ids, results), columns = ("PassengerId", "Survived",))
	return(dataFrame)

def toCSV(dataframe, filename):
	dataframe.to_csv(filename, index = False)