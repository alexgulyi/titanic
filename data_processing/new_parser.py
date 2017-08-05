from pandas import DataFrame, Series, read_csv, isnull
from sklearn.preprocessing import LabelEncoder
import statistics

class Parser(object):
	def __init__(self, featureStrategies):
		self.rawDataSet = DataFrame()
		self.dataSet = DataFrame()
		self.features = list(featureStrategies.keys())
		self.strategy = featureStrategies

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

	def parseColumn(self, name):
		col = self.getDataColumns(name)
		newCol = Series()
		strategy = self.strategy[name]
		if strategy['replace']:
			filledCol = col[~isnull(col)]
			replace = getattr(statistics, strategy['replace'])(filledCol) 
			newCol = col.fillna(replace)
		if strategy['encod']:
			enc = LabelEncoder()
			newCol = Series(enc.fit_transform(newCol))
		return(newCol)

	def fillMissings(self, name):

	def encodeFeature(self, name):

	def prepareDataSet(self):
		for f in self.features:
			print(f)
			self.dataSet[f] = self.parseColumn(f)

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
				"Rank"     : {'replace' : 'mode',   'encod' : True},
				"Sex"      : {'replace' : 'mode',   'encod' : True},
				"Embarked" : {'replace' : 'mode',   'encod' : True},
				"Ticket"   : {'replace' : 'median', 'encod' : False},}
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
	
	p.prepareDataSet()
	print(p.dataSet.head())