import numpy
from scipy import stats, optimize

if __name__ == '__main__':
	#dfTrain = prepareDataSet("train.csv")
	#dfTest = prepareDataSet("test.csv")
	#treeParams = {"max_depth" : 10, "min_samples_leaf" : 10, "min_impurity_split" : 0.1}
	target = "Survived"

	data = prepareDataSet("train.csv")
	dfTrain, dfTest = sampleData(data, 0.33)

	features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Rank", "Sex", "Embarked"]

	clfTree = createTree(dfTrain, features, target)
	
	#drawTree(clfTree, features)
	
	#result = clfTree.predict(dfTest.ix[:, dfTest.columns.difference(["PassengerId", "Survived"])])
	#score = metrics.accuracy_score(dfTest.loc[:, "Survived"], result)
	#print(score)

	#print(optimize.minimize(fun = lossFunction, x0 = (10, 2, 0.05), args = ('maxDepth', 'minSamplesLeaf', 'minImpuritySplit'), method = 'Nelder-Mead'))
	#predictions = dfTest["PassengerId"].to_frame()
	#predictions["Survived"] = Series(result)	
	#submission = predictions.to_csv('submission.csv', header = True, index = False)