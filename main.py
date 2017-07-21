import classify.tree as clftree
import data_processing.parser as prs
import data_processing.sampler as smp
import classify.optimize as opt
import config

if __name__ == '__main__':

	data = prs.prepareDataSet("train.csv")
	testData = prs.prepareDataSet("test.csv")

	clf = clftree.clfTree(config.features, config.target)

	optGini = opt.getOptGini(clf, data, config.samplingsNum, config.samplingsRate)

	clf.setImpurity(optGini[0])
	opt.logPredictionMetrics(clf, data, config.samplingsRate)

	#final prediction
	clf.fit(data)
	prediction = clf.predict(testData)
	prs.toCSV(prs.packResult(testData.loc[:, "PassengerId"], prediction['Y']), "submission.csv")