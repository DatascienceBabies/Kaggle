from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import Dataset

class Classifications:
    algorithms = {}

    def add_logistic_regression(self, cross_validation=5, rnd_state=0):
        lr = LogisticRegressionCV(cv=cross_validation, random_state=rnd_state)
        self.algorithms["Logistic Regression"] = lr

    def find_svm_linear_params(self, train_X, train_Y, num_steps=40, start=-3, stop=4):
        param_grid = {
            'C': np.logspace(start, stop, num_steps)
        }
        
        grid = GridSearchCV(svm.SVC(gamma='auto'), param_grid, cv=7)
        grid.fit(train_X, train_Y)
        print(grid.best_params_)
        return grid.best_params_

    def add_linear_svm(self, c):
        svc_linear = svm.SVC(kernel='linear', C=c)
        self.algorithms['Linear SVM'] = svc_linear        

    def find_svm_kernel_params(self, train_X, train_Y, num_steps=40, start=-3, stop=4, cv=7):
        param_grid = {
            'C': np.logspace(start, stop, num_steps),
            'gamma': np.logspace(start, stop, num_steps),
        }

        grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=cv)
        grid.fit(train_X, train_Y)

        print(grid.best_params_)
        return grid.best_params_

    def add_rbf_svm(self, c, gamma):
        svm_rbf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
        self.algorithms['RBF SVM'] = svm_rbf

    def add_naive_bayes(self):
        self.algorithms['Naive Bayes'] = GaussianNB()

    def add_classification_tree(self, depth=7):
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        self.algorithms['Classification Tree'] = clf

    def randomized_random_forest_parameter_search(self, train_X, train_Y):
        param_grid = {
            'max_features': ['auto', 'sqrt'],
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 7000, num = 10)],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
        }

        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        rf_random.fit(train_X, train_Y)

        print(rf_random.best_params_)
        return rf_random.best_params_

    def grid_search_for_params(self, train_X, train_Y, max_features, n_estimators, max_depth, min_sample_split, min_samples_leaf, bootstrap):
        param_grid = {
            'max_features': max_features,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_sample_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap,
        }

        rf = RandomForestRegressor()
        rf_gridCV = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, verbose=2, n_jobs = -1)
        rf_gridCV.fit(train_Y, train_Y)

        print(rf_gridCV.best_params_)
        return rf_gridCV.best_params_

    def add_random_forest(self, max_features='auto', max_depth=5, n_estimators=5000, min_samples_split=4, min_samples_leaf=1, bootstrap=True):
        rfc = RandomForestClassifier(max_features=max_features, max_depth=max_depth, n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap)
        self.algorithms['Random Forest'] = rfc

    def find_best_algorithm(self, train_x, train_y, test_x, test_y):
        best_score = 0

        for alg_name in self.algorithms:
            alg = self.algorithms[alg_name]
            alg.fit(train_x, train_y)
            score = alg.score(test_x, test_y)
            print("{0} scored {1}".format(alg_name, score))
            
            if score > best_score:
                best_score = score
                best_model_name = alg_name
                best_model = alg
            
        print('Best Model is: {0} with a score of {1} - will continue with this'.format(best_model_name, best_score))
        return best_model

