import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import glob
from keras.models import Sequential
from keras.layers import GRU,LSTM, SimpleRNN, Dense, Dropout
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# import missingno as msno
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import math
import time

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 10)


'''read data'''
def read_file(path):
    records = []
    f = glob.glob(path + '*.csv')
    for record in f:
        df = pd.read_csv(record)
        records.append(df)
    record_list = pd.concat(records)

    return record_list

data = read_file('csvs_per_year/')
cols = ['date', 'station', 'BEN', 'CH4', 'CO', 'EBE', 'MXY', 'NMHC', 'NO', 'NO_2', 'NOx', 'OXY',
       'O_3', 'PM10', 'PM25', 'PXY', 'SO_2', 'TCH', 'TOL']
data = data[cols]
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by =['date'])

'''plot the figures and check the missing values'''
# msno.matrix(data)
# plt.savefig('Particle.png')
# msno.bar(data)
# plt.savefig('Partical_bar.png')
# plt.close()

'''among the particles choose three most importante elements with less missing valules'''
''' plot an example'''
# def plot_example(station_id):
#     with pd.HDFStore('madrid.h5') as data:
#         example = data[station_id]
#         station_id = [k[1:] for k in data.keys() if k != '/master']
#         col_name = []
#         count_col = []
#         data_list = []
#         for i in station_id:
#             i = i.lstrip('/')
#             df = data[i]
#             df_copy = df.copy()
#             df_copy['station'] = i
#             data_list.append(df_copy)
#             df = df.sort_index()
#             col_name.append(df.columns)
#             msno.matrix(df)
#             plt.savefig('station_matrix/' + i + '.png')
#             plt.close()
#
#     fig, ax = plt.subplots(figsize=(20, 5))
#     examp_particles = example[['NO_2','O_3', 'PM10']]
#
#     examp_particles /= examp_particles.max(axis=0)
#
#     examp_particles.interpolate(method='time').rolling(window=24*15).mean().plot(ax=ax)
#     plt.savefig('example.png')
#     plt.show()
#     plt.close()
#
# plot_example('28079018')

chose_station =data[data['station'] ==28079018]
df = chose_station[['date','O_3']]
# print(df)
# print('==================')
# print()


'''data preprocess'''
'''I want to predict the next 24 hour air quality'''
'''so I will use 48 hours as a time node'''
'''which means, first 24 hours for input data, next 24 hours for output label'''

num_steps = 24
batch_size = 128
n_feature = 1
drop_out = 0.2

values = np.array(df.O_3)
ts = pd.Series(values, index=df.date)

def pivot_with_num_steps(series, num_steps):
    pivot = pd.DataFrame(index = df.date)

    for t in range(num_steps * 2):
        pivot['t_{}'.format(t)] = series.shift(-t)

    pivot = pivot.dropna(how='any')
    return pivot


series = (ts.interpolate(method='time')
                 .pipe(pivot_with_num_steps, num_steps))

series = series[(series.index.hour % 12) == 0]
if series.shape[0] % batch_size != 0:
    series = series.iloc[:-(series.shape[0]% batch_size)]
# print(series)


'''scaler the data'''
scaler = MinMaxScaler()
series = scaler.fit_transform(series)

# scaler = StandardScaler()
# series = scaler.fit_transform(series)

'''split the data to train and test'''

test_ratio = 0.2

split_point = int(series.shape[0] * (1 - test_ratio))
split_point -= split_point % batch_size

train_X = np.expand_dims(series[:split_point , :num_steps],axis=2)
train_y = np.expand_dims(series[:split_point, num_steps:],axis=2)
test_X = np.expand_dims(series[split_point:, :num_steps], axis=2)
test_y = np.expand_dims(series[split_point:, num_steps:],axis=2)
print('The size of input training size:')
print(train_X.shape)
print(train_y.shape)
print()
print('The size of inpurt test size: ')
print(test_X.shape)
print(test_y.shape)


'''build model'''

def generate_model(layer, neuron,batch_size, num_steps, feature, in_activation, out_activation, optimizer, loss):
    model= Sequential()
    model.add(layer(neuron[0],batch_input_shape=(batch_size, num_steps, feature),activation= in_activation,return_sequences=True))
    # model.add(Dense(neuron[1]))
    # model.add(Dropout(0.2))
    # model.add(Dense(neuron[1],  activation=out_activation))

    model.add(Dense(neurons[1]))
    model.add(Dropout(0.2))
    model.add(Dense(neurons[2]))
    model.add(Dropout(0.2))
    model.add(Dense(neuron[3], activation=out_activation))

    model.compile(loss=loss, optimizer=optimizer)
    return model

'''train the model'''

# optimizer = RMSprop(lr=0.005)
# optimizer = 'nadam'
optimizer = 'adam'
callback_early_stopping =EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
callback_reduce_lr = ReduceLROnPlateau(monitor= 'val_loss', factor = 0.1, min_lr = 1e-6, patience= 0, verbose=1)
callback_modelcheck = ModelCheckpoint('best_model.h5',monitor= 'val_loss',save_best_only=True, save_weights_only=False)
epochs = 50
neurons = (256,64,32,1)
input_activation ='tanh'
output_activation = 'sigmoid'
loss = 'mean_squared_error'


'''SimpleRNN'''

rnn_time = time.process_time()
simple_rnn = generate_model(SimpleRNN,neurons,batch_size,num_steps,n_feature,input_activation,output_activation,optimizer,loss)
print(simple_rnn.summary())

history_simple_rnn = simple_rnn.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,validation_data=(test_X, test_y),
              verbose=2, callbacks=[callback_early_stopping, callback_reduce_lr, callback_modelcheck])

rnn_time = time.process_time() - rnn_time

'''GRU'''
# gru_time = time.time()
gru_time = time.process_time()
gru = generate_model(GRU,neurons, batch_size,num_steps,n_feature,input_activation,output_activation, optimizer,loss)
print(gru.summary())

history_gru = gru.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,validation_data=(test_X, test_y),
              verbose=2, callbacks=[callback_early_stopping, callback_reduce_lr,callback_modelcheck])
gru_time = time.process_time() - gru_time

'''LTSM'''
lstm_time = time.process_time()
lstm = generate_model(LSTM,neurons,batch_size, num_steps,n_feature,input_activation,output_activation, optimizer,loss)
print(lstm.summary())
history_lstm = lstm.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,validation_data=(test_X, test_y),
                verbose=2, callbacks=[callback_early_stopping, callback_reduce_lr,callback_modelcheck])

lstm_time = time.process_time() - lstm_time


'''evaluate the model'''

def evaluation(model, histroy, layer):
    log = pd.DataFrame(histroy.history)
    print('The minimum loss of' ,layer, ': ', log.loc[log['loss'].idxmin]['loss'])
    print('The minimum validation loss of', layer, ': ', log.loc[log['val_loss'].idxmin]['val_loss'])

    trainPredict = model.predict(train_X, batch_size= batch_size)

    testPredict = model.predict(test_X, batch_size= batch_size)

    trainScore = model.evaluate(train_X, train_y, batch_size=batch_size,verbose=2)
    print(layer, 'Train scrore : %.3f MSE(%.3f RMSE) ' % (trainScore, math.sqrt((trainScore))))

    testScore = model.evaluate(test_X, test_y, batch_size=batch_size, verbose=2)
    print(layer, 'Test score : %.3f MSE (%.3f RMSE)' % (testScore, math.sqrt(testScore)) )

    r2_train_result = r2_score(train_y.reshape(train_y.shape[0],24), trainPredict.reshape(trainPredict.shape[0], 24))
    r2_test_result = r2_score(test_y.reshape((test_y.shape[0], 24)), testPredict.reshape((testPredict.shape[0], 24)))
    print(layer,'R2 train= ', r2_train_result)
    print(layer, 'R2 test =', r2_test_result)

evaluation(simple_rnn,history_simple_rnn, SimpleRNN)
print('Simple RNN time: %.2f' %(rnn_time/60))

evaluation(gru,history_gru, GRU)
print('GRU time: %.2f '% (gru_time/60))

evaluation(lstm, history_lstm, LSTM)
print('LSTM time: %.2f' % (lstm_time/60))
