import classify.tree as clftree
import data_processing.parser as prs
import data_processing.sampler as smp

if __name__ == '__main__':
	target = "Survived"
	features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Rank", "Sex", "Embarked"]
	samplingsNum = 6
	samplingsRate = 0.5

	data = prs.prepareDataSet("train.csv")
	testData = prs.prepareDataSet("test.csv")
	
	optima = []
	for i in range(1, samplingsNum):
		print("Sampling #{} on rate = {}:".format(i, samplingsRate))
		# making random train subsamples
		dfTrain, dfTest = smp.sampleData(data, samplingsRate)
		clf = clftree.clfTree(features, target)
		optimum = clf.fitOptimal(dfTrain, dfTest)
		optima.append(optimum)
		print("min(Gini) = {}\nmin(Error) = {}\n".format(*optimum))

	print("Choosen Gini value: {} deliveres minimal error = {}".format(*min(optima)))

	clf = clftree.clfTree(features, target)
	clf.setImpurity(min(optima)[0])
	clf.fit(data)
	prediction = clf.predict(testData)

	prs.toCSV(prs.packResult(testData.loc[:, "PassengerId"], prediction['Y']), "submission.csv")