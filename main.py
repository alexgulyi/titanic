import classify.tree as clftree
import data_processing.parser as prs
import data_processing.sampler as smp
import classify.optimize as opt

if __name__ == '__main__':
	target = "Survived"
	features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Rank", "Sex", "Embarked"]
	samplingsNum = 6
	samplingsRate = 0.5

	data = prs.prepareDataSet("train.csv")
	testData = prs.prepareDataSet("test.csv")
	clf = clftree.clfTree(features, target)

	optGini = opt.getOptGini(clf, data, samplingsNum, samplingsRate)

	clf.setImpurity(optGini)
	opt.validateModel(clf, data, samplingsRate)

	#final prediction
	clf.fit(data)
	prediction = clf.predict(testData)
	prs.toCSV(prs.packResult(testData.loc[:, "PassengerId"], prediction['Y']), "submission.csv")