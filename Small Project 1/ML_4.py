import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns

wine = datasets.load_wine()
X = wine.data[:, [0, 2]]
y = wine.target
print('Class labels:', np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

sns.set()
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

from sklearn.metrics import accuracy_score
# accuracy3 = []
# weights, params = [], []
# for c in np.arange(-4., 4.):
#     lr4 = LogisticRegression(C=10.**c, random_state=0)
#     lr4.fit(X_train_std, y_train)
#     weights.append(lr4.coef_[1])
#     params.append(10**c)
#     y_pred = lr4.predict(X_test_std)
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     accuracy3.append(accuracy_score(y_test, y_pred))
#
# plt.plot(params, accuracy3,color='r', linestyle='--',label='Predictor Accuracy')
accuracy1 = []
weights, params = [], []
for c in np.arange(-4., 4.):
    svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.**c)
    svm.fit(X_train_std, y_train)
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print("c ", c)
    print('Accuracy gamma=0.1 : %.2f' % accuracy_score(y_test, y_pred))
    accuracy1.append(accuracy_score(y_test, y_pred))

plt.plot(params, accuracy1,color='r', linestyle=':',label='Predictor Accuracy Gamma=0.1')

accuracy4 = []
weights, params = [], []
for c in np.arange(-4., 4.):
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print("c ", c)
    print('Accuracy gamma=10: %.2f' % accuracy_score(y_test, y_pred))
    accuracy4.append(accuracy_score(y_test, y_pred))

plt.plot(params, accuracy4,color='b', linestyle=':',label='Predictor Accuracy Gamma=10')

plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.title("SVM comparison")
plt.xscale('log')
plt.show()
