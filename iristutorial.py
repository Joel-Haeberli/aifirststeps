#Load Libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Data from url and read it with panda
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


## --- DATA ANALYSIS --- ##
#shape (form of the data -> size of the data in elements)
print(dataset.shape)

#head (read the 20 first values)
print(dataset.head(20))

#descriptions (get median, max, min, etc...)
print(dataset.describe())

# class distribution (count data sorted by column)
print(dataset.groupby('class').size())
## --- DATA ANALYSIS END --- ##


## --- DATA VISUALIZATION --- ##
#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
#histograms
dataset.hist()
plt.show()
#scatter plot matrix
scatter_matrix(dataset)
plt.show()
## --- DATA VISUALIZATION END --- ##


## --- EVALUATE ALGORITHM --- ##
#Split-out validation dataset (take 20% of the data as validation dataset to make sure later, that our tested algorithms are working properly. And perhaps we can decide which is the best one)
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state=seed)
#test options and evaluation metric
seed = 7
scoring = 'accuracy'
#Check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
#evaluate each model in turn
results = []
modelnames = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
## --- EVALUATE ALGORITHM END --- ##


## --- MAKE PREDICTIONS --- ##
## --- MAKE PREDICTIONS END --- ##

