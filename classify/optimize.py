from scipy.optimize import minimize

def lossFunction(tune, tree, testData):
		clf = tree
		clf.setTreeParams(tune)
		clf.fit(testData)
		return(1. - clf.predict(testData, crossValidate = True)['score'])

def minimizeFunction(function, tree, testData):
	print(minimize(fun = function,
					x0 = [5,10,5,0.20],
					args = (tree, testData,),
					bounds = ((3,10), (10,50), (5,20), (.01, .20)),
					method = 'SLSQP'))