"""

Script for training ML models

"""

## Libraries
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics  import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

## Load dataset
iris = datasets.load_iris()

## Select predictors and target variables
X = iris.data[:, [2,3]]
y = iris.target

## Split datsets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    random_state = 0)

## Scale and center dataset.
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

## Predict values
ppn = Perceptron(n_iter = 40,
                 eta0 = 0.1,
                 random_state = 0)
## Model train
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

## Acccuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

## Plot regions
## Plot decision boundaries
def plot_decision_regions(X, y,
                          classifier,
                          resolution=0.02,
                          test_idx=None,
                          x1label = 'x1', x2label = 'x2'):
    ## Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors  = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap    = ListedColormap(colors[:len(np.unique(y))])
    ## Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    ## Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
                    alpha = 0.8, c = cmap(idx),
                    marker = markers[idx], label = cl)
    plt.xlabel(x1label)
    plt.ylabel(x2label)

    # Highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c = '',
                    alpha = 1.0, linewidth = 1, marker = 'o',
                    s=55, label = 'test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined     = np.hstack((y_train, y_test))
plot_decision_regions(X = X_combined_std,
                      y = y_combined,
                      classifier = ppn,
                      test_idx = range(105, 150))
plt.show()
