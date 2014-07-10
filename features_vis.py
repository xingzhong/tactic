import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

if __name__ == '__main__':
	df = pd.read_csv('features.csv', names=['defence', 'degree', 'split'])
	df.plot(subplots=True)
	plt.show()