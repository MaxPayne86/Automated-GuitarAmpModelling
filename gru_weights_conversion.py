import random
import numpy as np
import tensorflow as tf

SEED=1995
hidden_size=5
input_size=5

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

gru_tf = tf.keras.layers.GRU(
    units=hidden_size,
    return_sequences=True,
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=SEED),
    bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
)

y_tf1 = gru_tf(tf.ones((1, 3, hidden_size)), training=False)  # forward pass with ones
print("\n\n")
print(y_tf1)

np.savez(
    'tf_gru_model_weights.npz',
    gru_kernel=gru_tf.weights[0].numpy(),
    gru_recurrent_kernel=gru_tf.weights[1].numpy(),
    gru_bias=gru_tf.weights[2].numpy()
)

import torch

torch.set_printoptions(precision=8)

torch.manual_seed(SEED)

npz_weights = np.load('tf_gru_model_weights.npz')

def convert_input_kernel(kernel):
    kernel_r, kernel_z, kernel_h = np.hsplit(kernel, 3)
    return np.concatenate((kernel_z.T, kernel_r.T, kernel_h.T))

def convert_recurrent_kernel(kernel):
    kernel_r, kernel_z, kernel_h = np.hsplit(kernel, 3)
    return np.concatenate((kernel_z.T, kernel_r.T, kernel_h.T))

def convert_bias(bias):
    bias = bias.reshape(2, 3, -1)
    return bias[:, [1, 0, 2], :].reshape((2, -1))

gru_torch = torch.nn.GRU(
    hidden_size=hidden_size,
    input_size=input_size,
    num_layers=1,
    bidirectional=False,
    batch_first=True
)

for pn, p in gru_torch.named_parameters():
    if 'weight_ih' in pn:
        p.data = torch.from_numpy(convert_input_kernel(npz_weights['gru_kernel']))
    elif 'weight_hh' in pn:
        p.data = torch.from_numpy(convert_recurrent_kernel(npz_weights['gru_recurrent_kernel']))
    elif 'bias_ih' in pn:
        p.data = torch.from_numpy(convert_bias(npz_weights['gru_bias'])[0])
    elif 'bias_hh' in pn:
        p.data = torch.from_numpy(convert_bias(npz_weights['gru_bias'])[1])

gru_torch.eval()
y_pt, _ = gru_torch(torch.ones(1, 3, hidden_size))
print(y_pt)
print("\n\n")

# Loss
np_tensor = y_pt.detach().numpy()
y_tf2 = tf.convert_to_tensor(np_tensor)
loss = tf.keras.losses.mean_squared_error(y_tf1.numpy(), y_tf2.numpy())
print("loss = \n%.8f\n" % np.sum(loss.numpy()))

# Back to keras

model_data = {}
model_state = gru_torch.state_dict()
for each in model_state:
    model_state[each] = model_state[each].tolist()
model_data['state_dict'] = model_state

WVals = np.array(model_data['state_dict']['weight_ih_l0'])
UVals = np.array(model_data['state_dict']['weight_hh_l0'])
bias_ih_l0 =  np.array(model_data['state_dict']['bias_ih_l0'])
bias_hh_l0 = np.array(model_data['state_dict']['bias_hh_l0'])

gru_weights = []

print(np.shape(WVals))
WVals = np.transpose(WVals)
WVals = WVals.reshape(hidden_size, 3, hidden_size)
for subm in WVals:
    subm[[0, 1]] = subm[[1, 0]]
WVals = WVals.reshape(hidden_size, 3*hidden_size)

gru_weights.append(WVals)

print(np.shape(UVals))
UVals = np.transpose(UVals)
UVals = UVals.reshape(hidden_size, 3, hidden_size)
for subm in UVals:
    subm[[0, 1]] = subm[[1, 0]]
UVals = UVals.reshape(hidden_size, 3*hidden_size)

gru_weights.append(UVals)

tmp = np.zeros((2, hidden_size*3))
tmp[0] = bias_ih_l0
tmp[1] = bias_hh_l0
tmp = tmp.reshape(2, 3, -1)
BVals = tmp[:, [1, 0, 2], :].reshape((2, -1))
gru_weights.append(BVals) # BVals is (2, hidden_size*3)
gru_tf = tf.keras.layers.GRU(units=hidden_size, return_sequences=True, weights=gru_weights)

y_tf3 = gru_tf(tf.ones((1, 3, hidden_size)), training=False)  # forward pass with ones
print(y_tf3)
print("\n\n")

# Loss
loss = tf.keras.losses.mean_squared_error(y_tf1.numpy(), y_tf3.numpy())
print(loss)
print("loss = \n%.8f\n" % np.sum(loss.numpy()))
