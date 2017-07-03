from scipy.optimize import minimize

def lossFunction(minImpurity, tree, testData):
		clf = tree
		clf.setImpurity(minImpurity)
		clf.fit(testData)
		return(1. - clf.predict(testData, crossValidate = True)['score'])

def minimizeFunction(function, tree, testData):
	print(minimize(fun = function,
					x0 = 0.5,
					args = (tree, testData,),
					# 'Nelder-Mead'
					method = 'SLSQP'))