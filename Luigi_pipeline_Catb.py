# удалить answers_test.csv и в терминале набрать команду: luigid

import pickle
#import numpy as np
import pandas as pd
import dask.dataframe as dd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
import luigi

#from my_pca import My_pca
#from my_functions import My_functions

from my_functions import my_merge, buy_time_change, get_offer_count, log_feature_1, \
    log_feature_2, log_feature_3, log_feature_4, target_encoder, log_columns, to_int
import my_pca

class MyCatBoostClassifier(CatBoostClassifier):
    def predict(self, X, threshold=0.79):
        result = super(MyCatBoostClassifier, self).predict_proba(X)
        predictions = [1 if p > threshold else 0 for p in result[:, 1]]
        return predictions

class LuigiPipeline(luigi.Task):
    TRAIN_FILENAME = luigi.Parameter()
    FEATURES_FILENAME = luigi.Parameter()
    #TEST_FILENAME = luigi.Parameter()

    def run(self):
        #from my_functions import My_functions
        data_train = pd.read_csv(self.TRAIN_FILENAME)
        data_train = data_train.drop(columns=['Unnamed: 0'])
        features = dd.read_csv(self.FEATURES_FILENAME, sep='\t', blocksize='300MB')

        train = my_merge(data_train.sample(50000), features)
        test = my_merge(data_train.sample(30000), features)

        step_2 = my_pca.My_pca(['vas_id'])
        train = step_2.fit_transform(train)
        test = step_2.transform(test)

        train = buy_time_change(train)
        test = buy_time_change(test)

        train_few = get_offer_count(train)
        test_few = get_offer_count(test)

        train = log_feature_1(train_few, train)
        test = log_feature_1(test_few, test)

        train = log_feature_2(train_few, train)
        test = log_feature_2(test_few, test)

        train = log_feature_3(train_few, train)
        test = log_feature_3(test_few, test)

        train = log_feature_4(train_few, train)
        test = log_feature_4(test_few, test)

        train.drop(columns=['date'], inplace=True)
        test.drop(columns=['date'], inplace=True)

        train, test = target_encoder(train, test, 'vas_id')

        test = log_columns(test, ['vas_id_mean'])
        test = to_int(test, ['vas_id'])

        with open('models/clf.pkl', 'rb') as model:
            clf = pickle.load(model)

        predicted = clf.predict(test.drop(columns=['target']))
        print('TEST\n\n' + classification_report(test['target'], predicted))
        pd.Series(predicted).to_csv('answers_test.csv', index=False)

    def output(self):
        return luigi.LocalTarget('answers_test.csv')

if __name__ == '__main__':

    #TRAIN_FILENAME = "current/data/data_train.csv"
    #FEATURES_FILENAME = "current/data/features.csv"
    #TEST_FILENAME = "current/data_test.csv"

    luigi.build([LuigiPipeline("data/data_train.csv", "data/features.csv")])