import pandas as pd
from sklearn.impute import KNNImputer

fileNames = ['NHIS_BENG_prototype.csv']
suffix = '_imputed'
ext = '.csv'

for fileName in fileNames:
    data = pd.read_csv(fileName + ext)
    columns_to_keep =  []
    for col in data.columns.values:
        # all the values for this feature are not null
        if data[col].notnull().sum() != 0:
            columns_to_keep.append(col)
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(data)
    imputed_data = pd.DataFrame(imputed_data, columns=columns_to_keep)
    imputed_data.to_csv(fileName + suffix + ext, index=False)
    print(f"Imputed data saved to '{fileName + suffix + ext}'.")