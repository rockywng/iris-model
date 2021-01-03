from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the required iris dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Stratified 10 fold cross validation will be used to estimate model accuracy
# In essence, our dataset will be split into ten parts, the model will be trained on 9 and will be 
# tested on 1
# 5 models will be tested: Logistic Regression (LR), K-Nearest Neighbours (KNN), 
# Classification and Regression Trees (CART), Gaussian 
# Niave Bayes (NB), Support Vector Machines (SVM)
# LR is a linear algorithm, the rest are non-linear

# Spot check the algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each algorithm
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# We can see that SVM has the highest accuracy score at about 98%

# Visual comparison of algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Visual Comparison of Algorithms')
pyplot.show()