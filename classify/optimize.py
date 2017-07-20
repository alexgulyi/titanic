from sklearn import metrics
import data_processing.sampler as smp


def getOptGini(tree, data, iterNum, sampleRate):
	optima = []
	for i in range(1, iterNum):
		print("Sampling #{} on rate = {}:".format(i, sampleRate))
		# making random train subsamples
		dfTrain, dfTest = smp.sampleData(data, sampleRate)
		optimum = tree.fitOptimal(dfTrain, dfTest)
		optima.append(optimum)
		print("min(Gini) = {}\nmin(Error) = {}\n".format(*optimum))
	opt = min(optima)
	print("Choosen Gini value: {} deliveres minimal error = {}".format(*opt))
	return(opt)

def validateModel(clf, data, sampleRate, log = True):
	"""Making cross-validated prediction and getting score, clf report and confusion matrix printed"""
	dfTrain, dfTest = smp.sampleData(data, sampleRate)
	clf.fit(dfTrain)
	result = clf.predict(dfTest, crossValidate = True)
	
	if log:
		print("Classification score = {}\n".format(result['score']))
		print(metrics.classification_report(dfTest.loc[:, 'Survived'], result['Y']))
		confMatrix = metrics.confusion_matrix(dfTest.loc[:, 'Survived'], result['Y']) / len(dfTest.index)
		print(confMatrix)
		print("True Negative: {}".format(confMatrix[0][0]))
		print("False Negative: {}".format(confMatrix[1][0]))
		print("True Positive: {}".format(confMatrix[0][1]))
		print("False Positive: {}".format(confMatrix[1][1]))