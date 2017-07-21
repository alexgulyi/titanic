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

def validateModel(clf, data, sampleRate, log = True):
	"""Making cross-validated prediction and getting score, clf report and confusion matrix printed"""
	dfTrain, dfTest = smp.sampleData(data, sampleRate)
	clf.fit(dfTrain)
	result = clf.predict(dfTest, crossValidate = True)
	
	if log:
		print("Classification score = {}\n".format(result['score']))
		print(metrics.classification_report(dfTest.loc[:, 'Survived'], result['Y']))
		# fix division
		confMatrix = metrics.confusion_matrix(dfTest.loc[:, 'Survived'], result['Y'])
		decFormat = "0:.2f"
		numNegative = list(result['Y']).count(0)
		numPositive = list(result['Y']).count(1)
		print(("True Negative: {" + decFormat + "}").format(confMatrix[0][0] / numNegative))
		print(("False Negative: {" + decFormat + "}").format(confMatrix[1][0] / numNegative))
		print(("True Positive: {" + decFormat + "}").format(confMatrix[0][1] / numPositive))
		print(("False Positive: {" + decFormat + "}").format(confMatrix[1][1] / numPositive))