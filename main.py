import classify.tree as clftree
import classify.optimize as opt
import data_processing.parser as prs
import data_processing.sampler as smp

from scipy.optimize import minimize


if __name__ == '__main__':
	target = "Survived"
	features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Rank", "Sex", "Embarked"]
	treeParameters = {'max_depth' : 10, 'min_samples_split' : 4, 'min_samples_leaf' : 10, 'min_impurity_split' : .10}

	data = prs.prepareDataSet("train.csv")
	dfTrain, dfTest = smp.sampleData(data, 0.33)	
	clf = clftree.clfTree(features, target)

	"""def lossFunction(max_depth, min_samples_split, min_samples_leaf, min_impurity_split):
		clf.setTreeParams([max_depth, min_samples_split, min_samples_leaf, min_impurity_split])
		clf.fit(dfTrain)
		return(1. - clf.predict(dfTest, crossValidate = True)['score']) """

	#print(lossFunction([10,4,10,0.1]))
	#print(minimize(fun = lossFunction, x0 = [10,5,10,0.5], args = ('max_depth', 'min_samples_split', 'min_samples_leaf', 'min_impurity_split'), method = 'Nelder-Mead'))

	"""clf.setParams(treeParameters)
	clf.fit(dfTrain)
	print(clf.predict(dfTest, crossValidate = True))"""