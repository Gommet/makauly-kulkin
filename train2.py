import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



group = sys.argv[1]

pickle = 'clean.pickle'

def get_df(pickle, train_ratio, is_train):
    df = pd.read_pickle(pickle)
    group_sales = df.groupby(level=0)['sales']
    df['new_sales'] = group_sales.shift(-7)
    df['sales4'] = group_sales.shift(4)
    df['sales7'] = group_sales.shift(7)
    df['sales10'] = group_sales.shift(10)
    df['sales14'] = group_sales.shift(14)
    df = df.dropna()
    df = df.loc[group].reset_index(drop=True)
    train_size = int(train_ratio * df.shape[0])
    df = df.iloc[:train_size] if is_train else df.iloc[train_size:].reset_index(drop=True)
    return df

df = get_df(pickle, 0.8, True)
scaler = MinMaxScaler()
scaler.fit(df)
df = scaler.transform(df)
print(df)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

