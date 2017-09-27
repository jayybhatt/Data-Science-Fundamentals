from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split


class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
    
    def score_summary(self, sort_by='mean_score'):
        def mean(numbers):
            return float(sum(numbers)) / max(len(numbers), 1)


        # def row(key, scores, params):
        #     d = {
        #          'estimator': key,
        #          'min_score': min(scores),
        #          'max_score': max(scores),
        #          'mean_score': mean(scores),
        #          # 'std_score': std(scores),
        #     }
        #     return pd.Series(dict(params.items() + d.items()))
                      
        # rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
        #              for k in self.keys
        #              for gsc in self.grid_searches[k].grid_scores_]
        # df = pd.concat(rows, axis=1)#.T.sort([sort_by], ascending=False)
        
        # columns = ['estimator', 'min_score', 'mean_score', 'max_score']#, 'std_score']
        # columns = columns + [c for c in df.columns if c not in columns]
        
        # return df[columns]
        for gsc in self.grid_searches[k].grid_scores_:

        return 

if __name__ == '__main__':

    models2 = { 
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        # 'GBR' : GradientBoostingRegressor()
    }

    params2 = { 
        'LinearRegression': { },
        'Ridge': { 'alpha': [0.1, 1.0] },
        'Lasso': { 'alpha': [0.1, 1.0] },
        # 'GBR' : {}
    }

    train_df = pd.read_csv("./train_data_mod.csv", index_col=0)
    train_y = train_df['logerror'].values
    # cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
    cat_cols = ["propertycountylandusecode", "propertyzoningdesc"]
    train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate']+cat_cols, axis=1)
    feat_names = train_df.columns.values


    X_train, X_valid, y_train, y_valid = train_test_split( train_df, train_y, test_size=0.3)

    helper2 = EstimatorSelectionHelper(models2, params2)
    helper2.fit(X_train, y_train, n_jobs=-1)

    print helper2.score_summary()
