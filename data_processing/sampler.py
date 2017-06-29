from pandas import DataFrame

def sampleData(dataframe, size):
	"""Splits dataframe randomly on 2 pieces with one sized <size>, 0 <= size <= 1"""
	df1 = dataframe.sample(frac = size)
	df2 = concat([dataframe, df1]).drop_duplicates(keep = False)
	return(df1, df2)
