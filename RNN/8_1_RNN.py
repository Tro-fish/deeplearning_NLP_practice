import numpy as np

timesteps = 10
input_dim = 4
hidden_units = 8

inputs = np.random.random((timesteps, input_dim))
hidden_state_t = np.zeros((hidden_units,)) 

Wx = np.random.random((hidden_units, input_dim))
Wh = np.random.random((hidden_units, hidden_units))
b = np.random.random((hidden_units,))

total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t))
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0)

print("모든 시점의 은닉 상태: \n", total_hidden_states)