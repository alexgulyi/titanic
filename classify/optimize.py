from sklearn import metrics
import data_processing.sampler as smp


def getMinGini(tree, data, iterNum, sampleRate):
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

def logPredictionMetrics(clf, data, sampleRate):
	"""Making cross-validated prediction and getting score, clf report and confusion matrix printed"""
	dfTrain, dfTest = smp.sampleData(data, sampleRate)
	clf.fit(dfTrain)
	result = clf.predict(dfTest, crossValidate = True)
	print("Classification score = {}\n".format(result['score']))
	print(metrics.classification_report(dfTest.loc[:, 'Survived'], result['Y']))
	print("Confusion matrix:")
	print(metrics.confusion_matrix(dfTest.loc[:, 'Survived'], result['Y']) / len(dfTest.index))