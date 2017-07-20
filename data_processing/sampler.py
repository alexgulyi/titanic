from pandas import DataFrame, concat

def sampleData(dataframe, size, intersect = False):
	"""Splits dataframe randomly on 2 pieces with one sized <size>, 0 <= size <= 1
	with empty intersection if intersect is True"""
	df1 = dataframe.sample(frac = size)
	
	if intersect:
		df2 = concat([dataframe, df1]).drop_duplicates(keep = False)
	else:
		# bootstrap-like sampling
		df2 = dataframe.sample(frac = size)
	
	return(df1, df2)
