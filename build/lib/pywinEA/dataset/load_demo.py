import pandas as pd
import numpy as np
import os
import pywinEA


def load_demo():
	"""
	Function that returns a dictionary with the values corresponding to the predictor variables,
	the labels and the names of the predictor variables in the form of numpy arrays.

		https://www.kaggle.com/uciml/breast-cancer-wisconsin-data#

	Attribute Information:

	1) ID number
	2) Diagnosis (1 = malignant, 0 = benign)
	3-32)

	Ten real-valued features are computed for each cell nucleus:

	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry
	j) fractal dimension ("coastline approximation" - 1)

	Creators:

	1. Dr. William H. Wolberg, General Surgery Dept.
	University of Wisconsin, Clinical Sciences Center
	Madison, WI 53792
	wolberg '@' eagle.surgery.wisc.edu

	2. W. Nick Street, Computer Sciences Dept.
	University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
	street '@' cs.wisc.edu 608-262-6619

	3. Olvi L. Mangasarian, Computer Sciences Dept.
	University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
	olvi '@' cs.wisc.edu

	Donor:

	Nick Street

	Returns
	----------
	:return (dict)
		Dataset.
	"""
	print("""
	Breast Cancer Wisconsin dataset. It contains a total of 569 samples of tumor and malignant cells. 
	Data labeled 1 corresponds to malignant cells, while data labeled 0 corresponds to benign cells. 
	The 30 characteristics contain real values obtained from images of cell nuclei. For more information:

			http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)


	The returned value is a dictionary where 'x_data' are the predictor variables, 'y_data' the class 
	labels and  'features' the name of the characteristics.
	""")
	path = '/'.join(os.path.abspath(pywinEA.__file__).split('/')[:-1])
	
	data = pd.read_csv(path+'/dataset/data/BreastCancerWisconsin.csv', index_col=0)
	x_data = data.iloc[:, 1:].values
	y_data = data.iloc[:, 0].values
	features = data.columns[1:].values

	# Transform labels
	y_data[np.where(y_data == 'M')] = 1
	y_data[np.where(y_data == 'B')] = 0
	y_data = y_data.astype(int)

	return {'x_data': x_data, 'y_data': y_data, 'features': features}