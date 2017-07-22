from sklearn import metrics
import data_processing.sampler as smp
from numpy import mean

def getOptGini(tree, data, iterNum, sampleRate, minError = True, bootstrap = False, log = True):
	"""Computes optimal Gini coefficient for a taken tree and data, sampled by a sampleRate."""
	optima = []
	
	for i in range(1, iterNum + 1):
		# making random train subsamples
		dfTrain, dfTest = smp.sampleData(data, sampleRate, intersect = bootstrap)
		optimum = tree.fitOptimal(dfTrain, dfTest)
		optima.append(optimum)
		if log:
			print("Sampling #{} (rate={}):".format(i, sampleRate) + " error={}, gini={}".format(*optimum))
	if minError:
		opt = min(optima)
		print("Min error rate = {} is delievered by gini = {}".format(*opt))
	else:
		errors = [ x[0] for x in optima ]
		ginies = [ x[1] for x in optima ]
		opt = (round(mean(errors),3), round(mean(ginies),3),)
		print("Average error rate = {} is delievered by gini = {}".format(*opt))
	return(opt)

def validateModel(clf, data, sampleRate, log = True):
	"""Making cross-validated prediction and getting score, clf report and confusion matrix printed"""
	dfTrain, dfTest = smp.sampleData(data, sampleRate)
	clf.fit(dfTrain)
	result = clf.predict(dfTest, crossValidate = True)

	if log:
		print("Estimated score = {}".format(round(result['score'], 3)))
		print("\nClassification report:")
		print(metrics.classification_report(dfTest.loc[:, 'Survived'], result['Y']))
		# fix division
		confMatrix = metrics.confusion_matrix(dfTest.loc[:, 'Survived'], result['Y'])
		decFormat = "0:.2f"
		numNegative = list(result['Y']).count(0)
		numPositive = list(result['Y']).count(1)
		print(("TN: {" + decFormat + "} FP: {" + decFormat + "}").format(confMatrix[0][0] / numNegative, confMatrix[0][1] / numNegative))
		print(("FN: {" + decFormat + "} TP: {" + decFormat + "}").format(confMatrix[1][0] / numNegative, confMatrix[1][1] / numNegative))

	# return false negative and false positive records
	dfTest["clfSurvived"] = result['Y']
	FN = dfTest[(dfTest["Survived"] == 1) & (dfTest["clfSurvived"] == 0)]
	FP = dfTest[(dfTest["Survived"] == 0) & (dfTest["clfSurvived"] == 1)]
	return(FN, FP)