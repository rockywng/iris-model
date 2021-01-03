from pandas import read_csv

# Load the required iris dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Find shape of dataset
# Provides the number of instances (rows) and the number of 
# attributes (columns) the data contains 
print(dataset.shape)

# Find head
# Prints the first 20 rows of data
print(dataset.head(20))

# Description of dataset
# Provides a summary of each attribute including the count, mean, min
# and max values
print(dataset.describe())

# Class distribution
# Lists out the classes found in the dataset and provides the number of
# instances of each class
print(dataset.groupby('class').size())