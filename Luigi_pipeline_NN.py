# удалить answers_test.csv и в терминале набрать команду: luigid

import pandas as pd
import dask.dataframe as dd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import luigi

import tensorflow_addons as tfa
from keras import models

from my_functions import my_merge, buy_time_change, get_offer_count, log_feature_1, \
    log_feature_2, log_feature_3, log_feature_4, target_encoder, log_columns, to_int, \
    prepare_example, get_dataset_from_csv

from my_pca import My_pca

class LuigiPipeline(luigi.Task):
    TRAIN_FILENAME = luigi.Parameter()
    FEATURES_FILENAME = luigi.Parameter()

    def run(self):
        #from my_functions import My_functions
        data_train = pd.read_csv(self.TRAIN_FILENAME)
        data_train = data_train.drop(columns=['Unnamed: 0'])
        features = dd.read_csv(self.FEATURES_FILENAME, sep='\t', blocksize='300MB')

        train = my_merge(data_train, features)
        test = my_merge(data_train, features)

        step_2 = My_pca(['vas_id'])
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
        train = log_columns(train, ['vas_id_mean'])
        train = to_int(train, ['vas_id'])

        train_data_columns = ['target', 'id', 'buy_time_train', 'vas_id', 'time_delta', 'component_1',
                              'component_3', 'month', 'day', 'weekofyear', 'how_old', 'novelty',
                              'vas_id_1', 'vas_id_2', 'vas_id_4', 'vas_id_5', 'vas_id_6', 'vas_id_7',
                              'vas_id_8', 'vas_id_9', 'vas_id_ord', 'vas_id_date_dif_1',
                              'vas_id_date_dif_2', 'vas_id_mean', 'log_vas_id_mean']

        train = train[train_data_columns]
        test = test[train_data_columns]

        scaler = MinMaxScaler()
        scaler.fit(train)
        test_data = pd.DataFrame(scaler.transform(test), columns=train_data_columns)

        test_data = test_data.astype('float32')
        test_data.target = test_data.target.map({1.0: 'accept', 0.0: 'reject'})

        CATEGORICAL_FEATURE_NAMES = ["vas_id"]

        test_data[CATEGORICAL_FEATURE_NAMES] = test_data[CATEGORICAL_FEATURE_NAMES].astype('str')

        test_data_file = "test_data.csv"
        test_data.to_csv(test_data_file, index=False, header=False)

        #with open("models/model_best_1", 'rb') as model:
        best_model = models.load_model("models/model_best_1")

        predicted = best_model.predict(get_dataset_from_csv(test_data_file))
        predicted = predicted.reshape((-1,))
        y_test = test_data['target']
        y_test = y_test.map(
            {'accept': 1, 'reject': 0}
        )
        print('TEST\n\n' + classification_report(y_test, (predicted>0.817)*1))
        pd.Series(predicted).to_csv('answers_test.csv', index=False)

    def output(self):
        return luigi.LocalTarget('answers_test.csv')

if __name__ == '__main__':

    luigi.build([LuigiPipeline("data/data_train.csv", "data/features.csv")])