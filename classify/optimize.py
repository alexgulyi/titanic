from sklearn import metrics
import data_processing.sampler as smp


def getOptGini(tree, data, iterNum, sampleRate, bootstrap = False):
	"""Computes optimal Gini coefficient for a taken tree and data, sampled by a sampleRate."""
	optima = []
	
	for i in range(1, iterNum):
		print("Sampling #{} on rate = {}:".format(i, sampleRate))
		# making random train subsamples
		dfTrain, dfTest = smp.sampleData(data, sampleRate, intersect = bootstrap)
		optimum = tree.fitOptimal(dfTrain, dfTest)
		optima.append(optimum)
		print("min(Gini) = {}\nmin(Error) = {}\n".format(*optimum))
	opt = min(optima)
	
	print("Choosen Gini value: {} deliveres minimal error = {}".format(*opt))
	
	return(opt)

def logPredictionMetrics(tree, data, sampleRate, bootstrap = False):
	"""Making cross-validated prediction and getting score, clf report and confusion matrix printed"""
	dfTrain, dfTest = smp.sampleData(data, sampleRate, intersect = bootstrap)
	tree.fit(dfTrain)
	result = tree.predict(dfTest, crossValidate = True)
	print("Classification score = {}\n".format(result['score']))
	print(metrics.classification_report(dfTest.loc[:, 'Survived'], result['Y']))
	print("Confusion matrix:")
	print(metrics.confusion_matrix(dfTest.loc[:, 'Survived'], result['Y']) / len(dfTest.index))