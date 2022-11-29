import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa

def my_merge(data_train, features):
    """Функция принимает исходные данные и получает из них требуемый рабочий датасет"""
    features_train = features.loc[features['id'].isin(data_train['id'])].compute()
    features_train = features_train.sort_values(by='buy_time')
    data_train = data_train.rename(columns={'buy_time': 'buy_time_train'})
    data_train = data_train.sort_values(by='buy_time_train')
    data_merged_train = pd.merge_asof(data_train, features_train, by='id', \
                                          left_on='buy_time_train', right_on='buy_time', direction='backward')
    data_merged_train_res = data_merged_train.dropna()
    return data_merged_train_res

def buy_time_change(X):
    """Функция преобразует Unix time в другие форматы и
    создает временные признаки"""
    from datetime import datetime, date

    X['time_delta'] = X['buy_time_train'] - X['buy_time']

    X['date'] = list(map(datetime.fromtimestamp, X['buy_time_train']))
    X['month'] = X['date'].apply(lambda x: x.timetuple()[1])
    X['day'] = X['date'].apply(lambda x: x.timetuple()[7])
    # неделя года
    X['weekofyear'] = X['buy_time_train'].apply(lambda x: pd.to_datetime(date.fromtimestamp(x)).weekofyear)
    X['time_max'] = X.buy_time_train.max()
    # Новизна предложения
    X['how_old'] = X['time_max'] - X['buy_time_train']
    X['novelty'] = 1 / (X['how_old'] + 1) * 100000
    X = X.drop(columns=['time_max', 'buy_time', 'component_2'])
    return X

def get_offer_count(df):
    """id пользователей для которых было несколько предложений"""
    b = pd.DataFrame(df['id'].value_counts())  # все id
    few = b.loc[b['id'] > 1]  # id,  у которых количество больше одного.
    return few

def log_feature_1(few, df):
    """история по всем предложениям.
    - если есть 3 предложения, т.е. в таблице 3 строки и в каждой будет храниться история про другие предложения, т.е.
    если РАЗНЫХ предложений 3, то в каждой (из трех) строк во всех трех столбцах будет стоять 1.
    - если 2 одинаковых предложения, то в каждой строке в соответствующем столбце будет стоять 2"""
    df['vas_id_1'] = 0  # для каждой услуги свой столбец
    df['vas_id_2'] = 0
    df['vas_id_4'] = 0
    df['vas_id_5'] = 0
    df['vas_id_6'] = 0
    df['vas_id_7'] = 0
    df['vas_id_8'] = 0
    df['vas_id_9'] = 0

    # цикл у нас только по тем пользователям у которых больше одного предложения
    # таких пользователей мы отобрали в предыдущей функции get_offer_count
    for i in few.index:
        # в целом конструкция аналогична df.loc[df['id']==i].groupby(['vas_id'])['vas_id'].count()
        a = df.loc[df['id'] == i, 'vas_id'].value_counts()
        # a = df.loc[df['id']==i].groupby(['vas_id'])['vas_id'].count()
        for k in a.index:
            df.loc[df['id'] == i, 'vas_id_' + str(int(k))] = a[k]
    return df

def log_feature_2(few, df):
    """В этой фиче порядковый номер для одинаковых предложений для одного пользователя, то есть
    если одному пользователю 2 раза предлагали
     одну и ту же услугу, то первая по времени услуга будет с номером 1, а вторая с номером 2"""
    df['vas_id_ord'] = 0

    # цикл у нас только по тем пользователям у которых больше одного предложения
    # таких пользователей мы отобрали в предыдущей функции get_offer_count
    for i in few.index:
        # тут получаем историю предложений для конкретного пользователя
        # историю мы сформировали ранее в функции log_feature_1
        a = df.loc[
            (df['id'] == i), ['vas_id_1', 'vas_id_2', 'vas_id_4', 'vas_id_5', 'vas_id_6', 'vas_id_7', 'vas_id_8',
                                  'vas_id_9']]
        # это мы получили номер столбца, он же номер услуги (vas_id: 1-9), в котором стоит больше 1,
        # т.е. предложение делалось клиенту более одного раза
        # т.е. если предложений 3, то в каждой строке хранится полная история и мы берем первую строку и те столбцы где >1
        aa = a.iloc[0][a.iloc[0].values > 1]
        # если есть такие столбцы, т.е. если пользователю предлагали одну услугу более одного раза
        # тут мы исходим из того, что ни одну услугу не предлагали 3 раза одному клиенту, максимум 2
        if len(aa) > 0:
            j = aa.index[0][7:]  # это мы взяли номер такой услуги, выдрали из названия столбца
            # отсортировали записи о предложении услуги по возрастанию даты
            a = df.loc[(df['id'] == i) & (df['vas_id'] == float(j))].sort_values(by='buy_time_train',
                                                                                     ascending=True)
            # для первой поставили номер 1 для второй номер 2
            for k in range(a.shape[0]):
                df.loc[a.index[k], 'vas_id_ord'] = k + 1
    return  df

def log_feature_3(few, df):
    """В этой фиче разница по датам для одинаковых предложений.
    Для первого предложения всегда 0, для второго +разница в днях+1 и т.д.
    Если предложения в один день, то между ними стоит 1 (разница в днях 0+1).
    Если оставить 0, то тогда не будет разницы между первыми и повторными предложениями"""

    df['vas_id_date_dif_1'] = 0

    # цикл у нас только по тем пользователям у которых больше одного предложения
    # таких пользователей мы отобрали в предыдущей функции get_proposal_count
    for i in few.index:
        # тут получаем историю предложений для конкретного пользователя
        # историю мы сформировали ранее в функции hist_columns_1
        a = df.loc[
            (df['id'] == i), ['vas_id_1', 'vas_id_2', 'vas_id_4', 'vas_id_5', 'vas_id_6', 'vas_id_7', 'vas_id_8',
                              'vas_id_9']]
        # это мы получили номер столбца, он же номер услуги (vas_id: 1-9), в котором стоит больше 1,
        # т.е. предложение делалось клиенту более одного раза
        # т.е. если предложений 3, то в каждой строке хранится полная история и мы бурум первую строку и те столбцы где >1
        aa = a.iloc[0][a.iloc[0].values > 1]
        # если есть такие столбцы, т.е. если пользователю предлагали одну услугу более одного раза
        # тут мы исходим из того, что ни одну услугу не предлагали 3 раза одному клиенту, максимум 2
        if len(aa) > 0:
            j = aa.index[0][7:]  # это мы взяли номер такой услуги, выдрали из названия столбца
            # эту услугу предлагали 2 раза, значит у нас есть столбец в котором стоит порядковый номер для такой услуги
            # мы его сделали ранее в функции hist_columns_2
               # поэтому первая услуга та, у которой в этом столбце 1, вторая - та, у которой 2.
            # находим разность по датам для этих услуг
            d1 = df.loc[(df['id'] == i) & (df['vas_id'] == float(j)) & (df['vas_id_ord'] == 1), 'date']
            d2 = df.loc[(df['id'] == i) & (df['vas_id'] == float(j)) & (df['vas_id_ord'] == 2), 'date']
            df.loc[(df['id'] == i) & (df['vas_id'] == float(j)) & (df['vas_id_ord'] == 2), 'vas_id_date_dif_1'] = \
                (pd.to_datetime(d2.values[0]) - pd.to_datetime(d1.values[0])).days + 1
    return df

def log_feature_4(few, df):
    """В этой фиче разница между всеми предложениями для одного клиента.
    Для первого предложения всега 0, для второго +разница в днях+1 и т.д.
    Если предложения в один день, то между ними стоит 1 (разница в днях 0+1).
    Если оставить 0, то тогда не будет разницы между первыми и повторными предложениями."""

    df['vas_id_date_dif_2'] = 0

    # цикл у нас только по тем пользователям у которых больше одного предложения
    # таких пользователей мы отобрали в предыдущей функции get_proposal_count
    for i in few.index:
        # получили все предложения пользователя и отсортировали все предложения по дате
        a = df.loc[(df['id'] == i)].sort_values(by='buy_time_train', ascending=True)
        # идем в цикле по предложениям начиная с 1го, не с 0го.
        for k in range(1, a.shape[0]):
            d1 = df.loc[a.index[k - 1], 'date']  # предыдущее предложение
            d2 = df.loc[a.index[k], 'date']  # первое предложение
            df.loc[a.index[k], 'vas_id_date_dif_2'] = (
                        pd.to_datetime(d2) - pd.to_datetime(d1)).days
            # + 1 этого не было, только здесь добавила
            # если предложений 3 и 2ое и 3е дали в один день (одновременно),
            # то у них разница с предыдущим предложением будет одинаковая
            # т.е. тут предыдущим предложением считаем первое предложение
            if (k == 2) & ((pd.to_datetime(d2) - pd.to_datetime(d1)).days == 0):
                df.loc[a.index[k], 'vas_id_date_dif_2'] = df.loc[a.index[k - 1], 'vas_id_date_dif_2']
    return df

def target_encoder(train, test, encoded):
    """Таргет энкодинг"""
    Mask = pd.DataFrame(train.groupby(by=encoded)['target'].mean()). \
        rename(columns={"target": f"{encoded}_mean"})
    train = pd.merge(train, Mask, how='left', on=encoded)
    test = pd.merge(test, Mask, how='left', on=encoded)
    return train, test

def log_columns(df, big_nunique_features):
    """Получение новых признаков как логарифм заданных признаков"""
    for col in big_nunique_features:
        const = np.min(df[col])
        if const > 0:
            const = 0
        else:
            const -= 0.1
        df[f"log_{col}"] = np.log(df[col]-const+0.001)
    return df

def to_int(df, cat_feats):
    """Функция для Катбуста"""
    df[cat_feats] = df[cat_feats].astype('int32')
    return df

CSV_HEADER = ['target', 'id', 'buy_time_train', 'vas_id', 'time_delta', 'component_1', 'component_3',
                      'month', 'day', 'weekofyear', 'how_old', 'novelty', 'vas_id_1', 'vas_id_2', 'vas_id_4',
                      'vas_id_5', 'vas_id_6', 'vas_id_7', 'vas_id_8', 'vas_id_9', 'vas_id_ord', 'vas_id_date_dif_1',
                      'vas_id_date_dif_2', 'vas_id_mean', 'log_vas_id_mean']
CATEGORICAL_FEATURE_NAMES = ["vas_id"]
NUMERIC_FEATURE_NAMES = [
    'id', 'time_delta', 'component_1', 'component_3', 'month', 'day',
    'weekofyear', 'how_old', 'novelty', 'vas_id_1', 'vas_id_2', 'vas_id_4', 'vas_id_5',
    'vas_id_6', 'vas_id_7', 'vas_id_8', 'vas_id_9', 'vas_id_ord', 'vas_id_date_dif_1',
    'vas_id_date_dif_2', 'vas_id_mean', 'log_vas_id_mean'
]

WEIGHT_COLUMN_NAME = "buy_time_train"
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
COLUMN_DEFAULTS = [
    [0.0] if feature_name in NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME] else ["NA"]
    for feature_name in CSV_HEADER
]
TARGET_FEATURE_NAME = "target"
TARGET_LABELS = ["reject", "accept"]

target_label_lookup = layers.StringLookup(
            vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
        )

def prepare_example(features, target):
    target_index = target_label_lookup(target)
    weights = features.pop(WEIGHT_COLUMN_NAME)
    return features, target_index, weights


def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        na_value="?",
        shuffle=shuffle,
    ).map(prepare_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return dataset.cache()

















