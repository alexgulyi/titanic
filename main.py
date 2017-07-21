import classify.tree as clftree
import data_processing.parser as prs
import data_processing.sampler as smp
import classify.optimize as opt
from config import target, features, samplingsNum, samplingsRate, bootstrapSampling

if __name__ == '__main__':

	data = prs.prepareDataSet("train.csv")
	testData = prs.prepareDataSet("test.csv")

	clf = clftree.clfTree(features, target)

	optGini = opt.getOptGini(clf, data, samplingsNum, samplingsRate)

	clf.setImpurity(optGini[0])
	opt.validateModel(clf, data, samplingsRate)

	#final prediction
	clf.fit(data)
	prediction = clf.predict(testData)
	prs.toCSV(prs.packResult(testData.loc[:, "PassengerId"], prediction['Y']), "submission.csv")