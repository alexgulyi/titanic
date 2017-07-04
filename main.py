import classify.tree as clftree
import data_processing.parser as prs
import data_processing.sampler as smp

if __name__ == '__main__':
	target = "Survived"
	features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Rank", "Sex", "Embarked"]

	data = prs.prepareDataSet("train.csv")
	dfTrain, dfTest = smp.sampleData(data, 0.33)
	clf = clftree.clfTree(features, target)
	
	def steady(tree, num = 100):
		"""Stabilize optimization process for a arbitrary tree"""
		solutions = []
		for i in range(0, num - 1):
			solutions.append(clf.optimize(dfTrain, dfTest))
		minError = min(solutions, key = lambda x : x[1])
		print(minError)
		return(solutions)
	
	print(steady(clf, 20))