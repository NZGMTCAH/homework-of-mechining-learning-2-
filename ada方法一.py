import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
# 请提前安装mlxtend，在命令行中输入pip install mlxtend
class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50
    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        Build the estimators for the AdaBoost estimator.
        '''
        sample_weight = np.ones(x.shape[0])/x.shape[0]
        for tree in range(self.n_estimator):
            estimator, sample_weight, estimator_weight= \
                self._boost(x,y, sample_weight)
            self.estimators_.append(estimator)
            self.estimator_weight_[tree]=estimator_weight



    def _boost(self, x, y, sample_weight):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array
        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)
        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        '''
        estimator = clone(self.base_estimator)
        dtc = estimator
        dtc.fit(x, y, sample_weight=sample_weight)
        pred_y = dtc.predict(x)
        indicator = np.ones(x.shape[0])*[pred_y!=y][0]
        err = np.dot(sample_weight, indicator) / np.sum(sample_weight)
        alpha = np.log((1-err)/err)
        new_sample_weight = sample_weight* np.exp(alpha*indicator)
        return estimator, new_sample_weight, alpha



    def predict(self, x):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''
        predicts = []
        for estimator in self.estimators_:
            pred = estimator.predict(x)
            pred[pred==0] = -1
            predicts.append(pred)

        predicts = np.array(predicts)

        pr = np.sign(np.dot(self.estimator_weight_, predicts))
        pr[pr==-1] = 0
        return pr


    def score(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''
        #accuracy = TP+TN / n
        predictions = self.predict(x)
        n= x.shape[0]
        tp = np.sum(predictions * y)
        tn = np.sum((1-predictions)* (1-y))
        acc = (tp+tn)/n
        return acc


    def sklearn_AdaBoostClassifier(self, x_train, y_train, x_test, y_test):
        model = AdaBoostClassifier(self.base_estimator, self.n_estimator)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)




def get_feature_range_mask(X, filler_feature_values=None,
                           filler_feature_ranges=None):
    if not isinstance(X, np.ndarray) or not len(X.shape) == 2:
        raise ValueError('X must be a 2D array')
    elif filler_feature_values is None:
        raise ValueError('filler_feature_values must not be None')
    elif filler_feature_ranges is None:
        raise ValueError('filler_feature_ranges must not be None')
    mask = np.ones(X.shape[0], dtype=bool)
    for feature_idx in filler_feature_ranges:
        feature_value = filler_feature_values[feature_idx]
        feature_width = filler_feature_ranges[feature_idx]
        upp_limit = feature_value + feature_width
        low_limit = feature_value - feature_width
        feature_mask = (X[:, feature_idx] > low_limit) & \
                       (X[:, feature_idx] < upp_limit)
        mask = mask & feature_mask
    return mask
def plot_decision_regions(X, y, clf,
                          feature_index=None,
                          filler_feature_values=None,
                          filler_feature_ranges=None,
                          ax=None,
                          X_highlight=None,
                          res=None,
                          zoom_factor=1.,
                          legend=1,
                          hide_spines=True,
                          markers='s^oxv<>',
                          colors=('#1f77b4,#ff7f0e,#3ca02c,#d62728,'
                                  '#9467bd,#8c564b,#e377c2,'
                                  '#7f7f7f,#bcbd22,#17becf'),
                          scatter_kwargs=None,
                          contourf_kwargs=None,
                          scatter_highlight_kwargs=None):
    from itertools import cycle
    import matplotlib.pyplot as plt
    import numpy as np
    from mlxtend.utils import check_Xy, format_kwarg_dictionaries
    import warnings
    from math import floor
    from math import ceil

    check_Xy(X, y, y_int=True)  # Validate X and y arrays
    dim = X.shape[1]

    if ax is None:
        ax = plt.gca()

    if res is not None:
        warnings.warn("The 'res' parameter has been deprecated."
                      "To increase the resolution, it's is recommended"
                      "to use to provide a `dpi argument via matplotlib,"
                      "e.g., `plt.figure(dpi=600)`.",
                      DeprecationWarning)

    plot_testdata = True
    if not isinstance(X_highlight, np.ndarray):
        if X_highlight is not None:
            raise ValueError('X_highlight must be a NumPy array or None')
        else:
            plot_testdata = False
    elif len(X_highlight.shape) < 2:
        raise ValueError('X_highlight must be a 2D array')

    if feature_index is not None:
        # Unpack and validate the feature_index values
        if dim == 1:
            raise ValueError(
                'feature_index requires more than one training feature')
        try:
            x_index, y_index = feature_index
        except ValueError:
            raise ValueError(
                'Unable to unpack feature_index. Make sure feature_index '
                'only has two dimensions.')
        try:
            X[:, x_index], X[:, y_index]
        except IndexError:
            raise IndexError(
                'feature_index values out of range. X.shape is {}, but '
                'feature_index is {}'.format(X.shape, feature_index))
    else:
        feature_index = (0, 1)
        x_index, y_index = feature_index

    # Extra input validation for higher number of training features
    if dim > 2:
        if filler_feature_values is None:
            raise ValueError('Filler values must be provided when '
                             'X has more than 2 training features.')

        if filler_feature_ranges is not None:
            if not set(filler_feature_values) == set(filler_feature_ranges):
                raise ValueError(
                    'filler_feature_values and filler_feature_ranges must '
                    'have the same keys')

        # Check that all columns in X are accounted for
        column_check = np.zeros(dim, dtype=bool)
        for idx in filler_feature_values:
            column_check[idx] = True
        for idx in feature_index:
            column_check[idx] = True
        if not all(column_check):
            missing_cols = np.argwhere(~column_check).flatten()
            raise ValueError(
                'Column(s) {} need to be accounted for in either '
                'feature_index or filler_feature_values'.format(missing_cols))

    marker_gen = cycle(list(markers))

    n_classes = np.unique(y).shape[0]
    colors = colors.split(',')
    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]

    # Get minimum and maximum
    x_min, x_max = (X[:, x_index].min() - 1. / zoom_factor,
                    X[:, x_index].max() + 1. / zoom_factor)
    if dim == 1:
        y_min, y_max = -1, 1
    else:
        y_min, y_max = (X[:, y_index].min() - 1. / zoom_factor,
                        X[:, y_index].max() + 1. / zoom_factor)

    xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
    xnum, ynum = floor(xnum), ceil(ynum)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=xnum),
                         np.linspace(y_min, y_max, num=ynum))

    if dim == 1:
        X_predict = np.array([xx.ravel()]).T
    else:
        X_grid = np.array([xx.ravel(), yy.ravel()]).T
        X_predict = np.zeros((X_grid.shape[0], dim))
        X_predict[:, x_index] = X_grid[:, 0]
        X_predict[:, y_index] = X_grid[:, 1]
        if dim > 2:
            for feature_idx in filler_feature_values:
                X_predict[:, feature_idx] = filler_feature_values[feature_idx]
    Z = clf.predict(X_predict.astype(X.dtype))
    Z = Z.reshape(xx.shape)
    # Plot decisoin region
    # Make sure contourf_kwargs has backwards compatible defaults
    contourf_kwargs_default = {'alpha': 0.45, 'antialiased': True}
    contourf_kwargs = format_kwarg_dictionaries(
        default_kwargs=contourf_kwargs_default,
        user_kwargs=contourf_kwargs,
        protected_keys=['colors', 'levels'])
    cset = ax.contourf(xx, yy, Z,
                       colors=colors,
                       levels=np.arange(Z.max() + 2) - 0.5,
                       **contourf_kwargs)

    ax.contour(xx, yy, Z, cset.levels,
               colors='k',
               linewidths=0.5,
               antialiased=True)

    ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])

    # Scatter training data samples
    # Make sure scatter_kwargs has backwards compatible defaults
    scatter_kwargs_default = {'alpha': 0.8, 'edgecolor': 'black'}
    scatter_kwargs = format_kwarg_dictionaries(
        default_kwargs=scatter_kwargs_default,
        user_kwargs=scatter_kwargs,
        protected_keys=['c', 'marker', 'label'])
    for idx, c in enumerate(np.unique(y)):
        if dim == 1:
            y_data = [0 for i in X[y == c]]
            x_data = X[y == c]
        elif dim == 2:
            y_data = X[y == c, y_index]
            x_data = X[y == c, x_index]
        elif dim > 2 and filler_feature_ranges is not None:
            class_mask = y == c
            feature_range_mask = get_feature_range_mask(
                X, filler_feature_values=filler_feature_values,
                filler_feature_ranges=filler_feature_ranges)
            y_data = X[class_mask & feature_range_mask, y_index]
            x_data = X[class_mask & feature_range_mask, x_index]
        else:
            continue

        ax.scatter(x=x_data,
                   y=y_data,
                   c=colors[idx],
                   marker=next(marker_gen),
                   label=c,
                   **scatter_kwargs)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if dim == 1:
        ax.axes.get_yaxis().set_ticks([])

    if plot_testdata:
        if dim == 1:
            x_data = X_highlight
            y_data = [0 for i in X_highlight]
        elif dim == 2:
            x_data = X_highlight[:, x_index]
            y_data = X_highlight[:, y_index]
        else:
            feature_range_mask = get_feature_range_mask(
                X_highlight, filler_feature_values=filler_feature_values,
                filler_feature_ranges=filler_feature_ranges)
            y_data = X_highlight[feature_range_mask, y_index]
            x_data = X_highlight[feature_range_mask, x_index]

        # Make sure scatter_highlight_kwargs backwards compatible defaults
        scatter_highlight_defaults = {'c': '',
                                      'edgecolor': 'black',
                                      'alpha': 1.0,
                                      'linewidths': 1,
                                      'marker': 'o',
                                      's': 80}
        scatter_highlight_kwargs = format_kwarg_dictionaries(
            default_kwargs=scatter_highlight_defaults,
            user_kwargs=scatter_highlight_kwargs)
        ax.scatter(x_data,
                   y_data,
                   **scatter_highlight_kwargs)

    if legend:
        if dim > 2 and filler_feature_ranges is None:
            pass
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels,
                      framealpha=0.3, scatterpoints=1, loc=legend)

    return ax

from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['label'] = iris.target #取标签
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label'] #数据特征
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# 数据划分
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)
from sklearn.model_selection import train_test_split  # 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
my_ada = AdaBoostBinaryClassifier(n_estimators=2, learning_rate=0.5)
my_ada.fit(X_train, y_train)
y_pred = my_ada.predict(X_test)
# print("my AdaBoost score", my_ada.score(X_test, y_test))
accuracy = 0.0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        accuracy += 1.0 / float(len(y_pred))
print(accuracy)
y_pre = my_ada.predict(X_test)

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plot_decision_regions(np.array(X_test), np.array(y_pre).astype('int64'), clf=my_ada, legend=2)
plt.xlabel('Length（cm）')
plt.ylabel('Width（cm）')
plt.title('AdaBoost (n_e=10, l_r=0.3)', fontsize=20)
plt.legend()  # 添加标签
plt.show()