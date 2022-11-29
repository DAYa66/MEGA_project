import pandas as pd
from sklearn.decomposition import PCA

class My_pca:

    def __init__(self, cat_feats):
        self.cat_feats = cat_feats
        self.scaler = None
        self.corr_feats = None
        self.not_corr_feats = ['target', 'id', 'buy_time_train', 'vas_id', 'buy_time']
        self.dim_reducer3d = None


    def convert_to_int(self, df):
        df[self.cat_feats] = df[self.cat_feats].astype('int8')

        return df

    def my_scaler_fit(self, df):
        """Обучаю RobustScaler для стандартизации нумерованных фичей для   упаковки в  PCA"""
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        self.scaler.fit(df[self.corr_feats])

        # return self.scaler

    def my_scaler_transform(self, df):
        """Преобразую RobustScaler_ом нумерованные фичи для упаковки в  PCA и показываю корреляцию этих фичей"""
        df_norm = self.scaler.fit_transform(df[self.corr_feats])

        df_1 = df[self.not_corr_feats]
        df_1 = df_1.reset_index(drop=True)
        df_norm = pd.DataFrame(df_norm, columns = self.corr_feats)
        df_scaled =pd.concat([df_1, df_norm], axis=1)

        return df_scaled

    def reduce_dims(self, df, dims=2, method='pca'):
        if method=='pca':
            dim_reducer = PCA(n_components=dims, random_state=42)
            components = dim_reducer.fit_transform(df)
        else:
            print('Error')

        colnames = ['component_' + str(i) for i in range(1, dims +1)]
        return dim_reducer, pd.DataFrame(data = components, columns = colnames)

    def pca_fit_transform(self, train):
        """Функция сжимающая малозначимые признаки методом PCA и возвращающая кроме
        обработанного датасета натренированную модель PCA"""
        self.dim_reducer3d, components_3d = self.reduce_dims(train[self.corr_feats],
                                                             dims=3, method='pca')
        train_pca = pd.concat([train[self.not_corr_feats], components_3d], axis=1)

        return train_pca

    def pca_transform(self, test):
        """Функция сжимающая малозначимые признаки методом PCA на тестовой выборке"""
        dims = 3
        components = self.dim_reducer3d.transform(test[self.corr_feats])
        colnames = ['component_' + str(i) for i in range(1, dims +1)]
        components_3d_test = pd.DataFrame(data = components, columns = colnames)
        test_pca = pd.concat([test[self.not_corr_feats],
                              components_3d_test], axis=1)

        return test_pca


    def fit_transform(self, train):
        train = self.convert_to_int(train)
        all_feats = train.columns.tolist()
        self.corr_feats = [i for i in all_feats if i not in self.not_corr_feats]
        self.my_scaler_fit(train)
        train = self.my_scaler_transform(train)
        train_pca = self.pca_fit_transform(train)
        return train_pca

    def transform(self, test):
        test = self.convert_to_int(test)
        test = self.my_scaler_transform(test)
        test_pca = self.pca_transform(test)
        return test_pca