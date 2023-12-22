import numpy as np
from keras.layers import LSTM, SimpleRNN, Bidirectional

# 단어 벡터의 차원이 5, 문장의 길이가 4
train_x = [[[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5,2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]]
train_x = np.array(train_x, dtype=np.float32)
print(train_x.shape) # (batch size, time steps, input dim)


lstm = LSTM(3, return_sequences=True, return_state=True)
hidden_states, last_hidden_state, last_cell_state = lstm(train_x)
print( 'hidden states : {}, shape: {}'. format(hidden_states, hidden_states.shape))
print('last hidden state : {}, shape: {}'. format(last_hidden_state,last_hidden_state.shape) )
print('last cell state : {}, shape: {}'. format(last_cell_state, last_cell_state.shape))