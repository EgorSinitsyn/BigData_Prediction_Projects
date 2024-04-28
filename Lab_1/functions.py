def prepare_X(df):
    # Переводит все имена столбцов в нижний регистр + замена пробелов на "_"
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Столбцы для удаления:
    columns_to_drop = ["datecrawled", "lastseen;;;;;;;;", "datecreated", "name", "nrofpictures", "seller", "offertype"]
    df.drop(columns=columns_to_drop, inplace=True)

    # Заменить пропущенные значения в колонках "kilometer" и "nrofpictures" на 0
    df['kilometer'].fillna(0, inplace=True)

    df['yearofregistration'] = 2024 - df.yearofregistration
    df = df[df['yearofregistration'] >= 0]

    # kilometer, price, yearofregistration, monthofregistration, powerps, postalcode- меняем тип данных на числовой
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['yearofregistration'] = pd.to_numeric(df['yearofregistration'], errors='coerce')
    df['monthofregistration'] = pd.to_numeric(df['monthofregistration'], errors='coerce')
    df['kilometer'] = pd.to_numeric(df['kilometer'], errors='coerce')
    df['powerps'] = pd.to_numeric(df['powerps'], errors='coerce')
    df['postalcode'] = pd.to_numeric(df['postalcode'], errors='coerce')

    df.dropna(subset=['vehicletype', 'gearbox', 'model', 'fueltype', 'notrepaireddamage'], inplace=True)

    df['notrepaireddamage'] = df['notrepaireddamage'].map({'nein': 0, 'ja': 1})
    return df


def One_Hot_Encoding(df, cat_columns):
    one_hot = pd.get_dummies(df[cat_columns])
    df = df.drop(cat_columns, axis=1)
    df = pd.concat([df, one_hot], axis=1)
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