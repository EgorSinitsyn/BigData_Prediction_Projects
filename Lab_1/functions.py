def prepare_X(df_orig):
    # Переводит все имена столбцов в нижний регистр + замена пробелов на "_"
    df_orig.columns = df_orig.columns.str.lower().str.replace(' ', '_')

    # Столбцы для удаления:
    columns_to_drop = ["datecrawled", "lastseen;;;;;;;;", "datecreated", "name", "nrofpictures", "seller", "offertype"]
    df_orig.drop(columns=columns_to_drop, inplace=True)

    # Заменить пропущенные значения в колонках "kilometer" и "nrofpictures" на 0
    df_orig['kilometer'].fillna(0, inplace=True)

    df_orig['yearofregistration'] = 2024 - df_orig.yearofregistration
    df_orig = df_orig[df_orig['yearofregistration'] >= 0]

    # kilometer, price, yearofregistration, monthofregistration, powerps, postalcode- меняем тип данных на числовой
    df_orig['price'] = pd.to_numeric(df_orig['price'], errors='coerce')
    df_orig['yearofregistration'] = pd.to_numeric(df_orig['yearofregistration'], errors='coerce')
    df_orig['monthofregistration'] = pd.to_numeric(df_orig['monthofregistration'], errors='coerce')
    df_orig['kilometer'] = pd.to_numeric(df_orig['kilometer'], errors='coerce')
    df_orig['powerps'] = pd.to_numeric(df_orig['powerps'], errors='coerce')
    df_orig['postalcode'] = pd.to_numeric(df_orig['postalcode'], errors='coerce')

    df_orig.dropna(subset=['vehicletype', 'gearbox', 'model', 'fueltype', 'notrepaireddamage'], inplace=True)

    df_orig['notrepaireddamage'] = df_orig['notrepaireddamage'].map({'nein': 0, 'ja': 1})
    return df_orig


def One_Hot_Encoding(df, cat_columns):
    one_hot = pd.get_dummies(df[cat_columns])
    df = df.drop(cat_columns, axis=1)
    df = pd.concat([df, one_hot], axis=1)
    bool_columns = df.select_dtypes(include=[bool]).columns
    df[bool_columns] = df[bool_columns].astype(int)
    return df



class BinaryEncoder:
    def __init__(self, columns):
        self.columns = columns
        self.binary_mapping = {}

    def fit(self, data):
        for col in self.columns:
            unique_values = data[col].unique()
            binary_repr_len = len(bin(len(unique_values) - 1)[2:])
            self.binary_mapping[col] = {
                value: format(index, '0{}b'.format(binary_repr_len))
                for index, value in enumerate(unique_values)
            }

    def transform(self, data):
        encoded_data = data.copy()
        for col, mapping in self.binary_mapping.items():
            encoded_data[col] = encoded_data[col].map(mapping)
        return encoded_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)



def align_columns(big_df, small_df):
    # Создаем копию маленького датафрейма
    aligned_df = small_df.copy()

    # Получаем список столбцов из большого датафрейма
    big_columns = big_df.columns

    # Перебираем столбцы из большого датафрейма
    for col in big_columns:
        # Если столбец отсутствует в маленьком датафрейме, добавляем его в нужном порядке
        if col not in small_df.columns:
            aligned_df[col] = 0

    # Сортируем столбцы маленького датафрейма в соответствии с порядком столбцов большого датафрейма
    aligned_df = aligned_df[big_columns]

    # Возвращаем выровненный датафрейм
    return aligned_df