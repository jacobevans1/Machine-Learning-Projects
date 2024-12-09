import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

df = pd.read_csv('wine.csv', header=None)

print(df.tail())
#
# X = df.data[:, [2, 3]]
# y = df.target
# print('Class labels:', np.unique(y))

# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica




wine = datasets.load_wine()
# print(wine.tail())
X = wine.data[:, [0, 2]]
y = wine.target
print('Class labels:', np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)




from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


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
                    c='yellow',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')




from sklearn.linear_model import LogisticRegression

C1=[10, 100,1000,5000]


#10
#
lr = LogisticRegression(C=10.0, random_state=0)
lr.fit(X_train_std, y_train)


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')

plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=10")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()
#
# # 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
from sklearn.metrics import accuracy_score
# y_pred = lr.predict(X_test_std)
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
lr.predict_proba(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
    y_pred = lr.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()

#
#
# #100
#
lr = LogisticRegression(C=100.0, random_state=0)
lr.fit(X_train_std, y_train)


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')

plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.legend(loc='upper left')
plt.title("C=100")
plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()
#
# # 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
from sklearn.metrics import accuracy_score
# y_pred = lr.predict(X_test_std)
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
lr.predict_proba(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
    y_pred = lr.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()

#
#
# #1000
#
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')

plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=1000")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()
#
# # 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
from sklearn.metrics import accuracy_score
# y_pred = lr.predict(X_test_std)
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
lr.predict_proba(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
    y_pred = lr.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()


lr = LogisticRegression(C=5000.0, random_state=0)
lr.fit(X_train_std, y_train)


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')

plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=5000")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()


from sklearn.metrics import accuracy_score
# y_pred = lr.predict(X_test_std)
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
lr.predict_proba(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
    y_pred = lr.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()







# from sklearn.metrics import classification_report
# y_true = [0, 1, 2, 2, 2]
# y_pred = [0, 0, 2, 2, 1]
# target_names = ['class 0', 'class 1', 'class 2']
# print(classification_report(y_true, y_pred, target_names=target_names))



from sklearn.svm import SVC
#
#
# #SVM 10
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=10, gamma=0.1")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_wine_1.png', dpi=300)
plt.show()


accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
svm.predict(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    # lr = LogisticRegression(C=10.**c, random_state=0)
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    # weights.append(svm.coef_[1])
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()





# #SVM 100
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=100.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=100, gamma=0.1")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_wine_1.png', dpi=300)
plt.show()


accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
svm.predict(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    # lr = LogisticRegression(C=10.**c, random_state=0)
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    # weights.append(svm.coef_[1])
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()





# #SVM 1000
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=1000.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=1000, gamma=0.1")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_wine_1.png', dpi=300)
plt.show()


accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
svm.predict(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    # lr = LogisticRegression(C=10.**c, random_state=0)
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    # weights.append(svm.coef_[1])
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()





# #SVM 5000
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=5000.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=5000, gamma=0.1")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_wine_1.png', dpi=300)
plt.show()

accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
svm.predict(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    # lr = LogisticRegression(C=10.**c, random_state=0)
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    # weights.append(svm.coef_[1])
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()





# #SVM 10
svm = SVC(kernel='linear', random_state=0, gamma=10.0, C=10.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=10, gamma=10.0")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_wine_1.png', dpi=300)
plt.show()


accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
svm.predict(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    # lr = LogisticRegression(C=10.**c, random_state=0)
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    # weights.append(svm.coef_[1])
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()






# #SVM 100
svm = SVC(kernel='linear', random_state=0, gamma=10, C=100.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=100, gamma=10.0")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_wine_1.png', dpi=300)
plt.show()
accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
svm.predict(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    # lr = LogisticRegression(C=10.**c, random_state=0)
    svm = SVC(kernel='linear', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    # weights.append(svm.coef_[1])
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()




# #SVM 1000
svm = SVC(kernel='linear', random_state=0, gamma=10, C=1000.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=1000, gamma=10.0")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_wine_1.png', dpi=300)
plt.show()

accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
svm.predict(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    # lr = LogisticRegression(C=10.**c, random_state=0)
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    # weights.append(svm.coef_[1])
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()





# #SVM 5000
svm = SVC(kernel='linear', random_state=0, gamma=10, C=5000.0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.title("C=5000, gamma=10.0")
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_wine_1.png', dpi=300)
plt.show()

accuracy = []
# 3) Plot the accuracy_score (in sklearns) of your predictor with respect to
svm.predict(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    # lr = LogisticRegression(C=10.**c, random_state=0)
    svm = SVC(kernel='rbf', random_state=0, gamma=10, C=10.**c)
    svm.fit(X_train_std, y_train)
    # weights.append(svm.coef_[1])
    params.append(10**c)
    y_pred = svm.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
# plt.plot(params, weights[:, 0],label='sepal length')
# plt.plot(params, weights[:, 1], linestyle='--',label='petal length')
plt.plot(params, accuracy, linestyle='--',label='acid')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()


#XOR
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)


plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/xor.png', dpi=300)
plt.show()



svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_14.png', dpi=300)
plt.show()

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_15.png', dpi=300)
plt.show()

from sklearn.svm import SVC

svm = SVC(kernel='linear', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('alcohol [standardized]')
plt.ylabel('acid [standardized]')
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('images/03_15.png', dpi=300)
plt.show()

