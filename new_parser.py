from pandas import DataFrame, Series, read_csv, isnull, to_numeric
from sklearn.preprocessing import LabelEncoder
import statistics
from re import search

class Parser(object):
	def __init__(self, featureStrategies, factors):
		self.rawDataSet = DataFrame()
		self.dataSet = DataFrame()
		self.features = list(featureStrategies.keys())
		self.strategy = featureStrategies
		self.factors = factors
	
	def getDataColumns(self, names, origin = 'raw'):
		"""Args: names - iterable with column names
				 origin = 'raw' (returns from rawDataSet),
				 		  'prep' (returns from dataSet)"""
		if origin == 'raw' and not self.rawDataSet.empty:
			return(self.rawDataSet.loc[:, names])
		elif origin == 'prep' and not self.dataSet.empty:
			return(self.dataSet.loc[:, names])
		else:
			return(Series())

	def loadData(self, path):
		try:
			self.rawDataSet = read_csv(path)
		except OSError:
			pass

	def extractFromColumn(self, pattern, name, origin = 'raw'):
		result = []
		for value in self.getDataColumns(name, origin):
			try:
				match = search(pattern, value)
				result.append(match.group())
			except AttributeError:
				result.append(None)
				continue
		return(Series(result))


	def parseColumn(self, name):
		col = self.getDataColumns(name)
		newCol = Series()
		strategy = self.strategy[name]
		if strategy['replace']:
			filledCol = col[~isnull(col)]
			try:
				replace = getattr(statistics, strategy['replace'])(filledCol)
			except:
				# handling multiple modes
				if strategy['replace'] == 'mode':
					replace = max(set(filledCol), key = list(filledCol).count)
				else:
					replace = None
			newCol = col.fillna(replace)
		if strategy['encod']:
			enc = LabelEncoder()
			if self.factors[name]:
				enc.fit(self.factors[name])
			newCol = Series(enc.fit_transform(newCol))
			print(enc.classes_)
		return(newCol)

	def prepareDataSet(self):
		for f in self.features:
			print(f)
			self.dataSet[f] = self.parseColumn(f)

	def dataToCSV(self, data, filename):
		getattr(self, data).to_csv(filename, index = False)


features = ["Pclass",
			"Age",
			"SibSp",
			"Parch",
			"Fare",
			"Rank",
			"Sex",
			"Embarked",
			"Ticket"]

if __name__ == '__main__':
	strategy = {"Pclass"   : {'replace' : 'median', 'encod' : False},
				"Age"      : {'replace' : 'mean',   'encod' : False},
				"SibSp"    : {'replace' : 'median', 'encod' : False},
				"Parch"    : {'replace' : 'median', 'encod' : False},
				"Fare"     : {'replace' : 'mean',   'encod' : False},
				"Title"    : {'replace' : 'mode',   'encod' : True},
				"Sex"      : {'replace' : 'mode',   'encod' : True},
				"Embarked" : {'replace' : 'mode',   'encod' : True},
				"Ticket"   : {'replace' : 'mode', 'encod' : False},}

	factors = {'Title' : {['Mrs', 'Mr', 'Miss', 'Mlle', 'Ms', 'Mme',] : 0,
						  ['Major', 'Lady', 'the Countess', 'Don',  'Col', 'Rev', 'Dona', 'Sir', 'Capt', 'Jonkheer', 'Dr', 'Master'] : 1},
			   'Embarked' : {'Q' : 0, 'C' : 1, 'S' : 2},
			   'Sex' : {'female' : 0, 'male' : 1},
			  }

	p = Parser(strategy)
	
	print("Init tests")
	print(p.rawDataSet)
	print(p.dataSet)
	print(p.strategy == strategy)
	print(set(p.features) == set(features))
	
	print("Loading tests")
	p.loadData('train.csv')
	print(p.rawDataSet.head())
	print(p.dataSet.head())
	
	print("Getting columns tests")
	print(p.getDataColumns(['Pclass', 'Age']).head())
	print(p.getDataColumns(['Pclass', 'Age', 'Fareeeee']).head())
	print(p.getDataColumns(['Pclass', 'Age',], origin = 'prepared').head())

	print("Parsing columns tests")

	print(p.parseColumn('Embarked').head())
	print(set(p.parseColumn('Embarked')))
	print(set(p.getDataColumns('Embarked')))

	print(p.parseColumn('Age').head())
	
	print(p.getDataColumns('Name').head())

	p.rawDataSet['Title'] = p.extractFromColumn("(?<=,\s)([^\.])*", 'Name')
	p.rawDataSet['Name'] = p.extractFromColumn("^([^,])*", 'Name')
	p.rawDataSet['Ticket'] = p.extractFromColumn("\d+", 'Ticket')

	p.prepareDataSet()
	print(p.dataSet.head())

	p.dataToCSV('dataSet', 'train1.csv')

	p1 = Parser(strategy)
	p1.loadData('test.csv')
	
	p1.rawDataSet['Title'] = p1.extractFromColumn("(?<=,\s)([^\.])*", 'Name')
	p1.rawDataSet['Name'] = p1.extractFromColumn("^([^,])*", 'Name')
	p1.rawDataSet['Ticket'] = p1.extractFromColumn("\d+", 'Ticket')

	print(p1.rawDataSet.head())
	p1.prepareDataSet()
	p1.dataToCSV('dataSet', 'test1.csv')