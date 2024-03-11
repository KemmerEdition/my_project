import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from catboost import CatBoostRanker, Pool
from copy import deepcopy

train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')

qs = np.unique(train_df['search_id'].values)
q_train, q_val = train_test_split(qs, test_size=0.2, random_state=13)
match_q_train, match_q_test = (train_df['search_id'].isin(q_train)), (train_df['search_id'].isin(q_val))
fin_X_train, fin_X_val = train_df[match_q_train], train_df[match_q_test]

q_train = fin_X_train['search_id'].values
y_train = fin_X_train['target'].values
x_train = fin_X_train.drop(columns=['search_id', 'target']).values

q_val = fin_X_val['search_id'].values
y_val = fin_X_val['target'].values
x_val = fin_X_val.drop(columns=['search_id', 'target']).values

y_test = test_df['target'].values
q_test = test_df['search_id'].values
x_test = test_df.drop(columns=['search_id', 'target']).values


def select_features(x_train, y_train, x_test):
    fs = SelectFromModel(RandomForestClassifier(n_estimators=100, max_features=17))
    fs.fit(x_train, y_train)
    X_train_fs = fs.transform(x_train)
    X_test_fs = fs.transform(x_test)
    return X_train_fs, X_test_fs, fs


x_train_select, x_test_select, fs = select_features(x_train, y_train, x_test)
train_pool = Pool(data=x_train_select, label=y_train, group_id=q_train)
test_pool = Pool(data=x_test_select, label=y_test, group_id=q_test)
default_parameters = {
    'iterations': 2000,
    'custom_metric': ['NDCG','AverageGain:top=10'],
    'verbose': False,
    'random_seed': 13}


def fit_model(loss_function, additional_params=None, train_pool=train_pool, test_pool=test_pool):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model


my_model = fit_model('YetiRank')
metric_test = my_model.score(x_test_select, y_test, group_id=q_test, top=len(y_test))
print(f"NDCG-test (CatBoostRanker): {metric_test}")
