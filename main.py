import classify.tree as clftree
import classify.optimize as opt
import data_processing.parser as prs
import data_processing.sampler as smp

if __name__ == '__main__':
	target = "Survived"
	features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Rank", "Sex", "Embarked"]
	treeParameters = {'max_depth' : 10, 'min_samples_split' : 4, 'min_samples_leaf' : 10, 'min_impurity_split' : .10}

	data = prs.prepareDataSet("train.csv")
	dfTrain, dfTest = smp.sampleData(data, 0.33)	
	clf = clftree.clfTree(features, target)

	#for x in range(1,100):
	#	print(x / 100, opt.lossFunction(x / 100, clf, dfTest))
	#opt.minimizeFunction(opt.lossFunction, clf, dfTest)