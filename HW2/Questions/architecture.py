import pickle
import gzip
import urllib.request
from sklearn.preprocessing import scale
from mydnn import MyDNN
from Macros import generate_layer, plot_graphs, one_hot
import pickle
import os

log_path = '..\\LOG'
print(log_path)
# data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
# urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
training_samples = scale(train_set[0], axis=0, with_std=False)
validation_samples = scale(valid_set[0], axis=0, with_std=False)
training_classifications = one_hot(train_set[1])
validation_classifications = one_hot(valid_set[1])
num_samples, num_pixels = training_samples.shape

arch = {}
for depht in [3]:
    for neurones_l1 in [128, 254, 512]:
        input_size = num_pixels
        neurones = neurones_l1
        leyers = []
        for index in range(depht):
            layer = generate_layer(input_size, neurones, 'relu', "l2", 0.2)
            input_size = int(neurones)
            neurones = int((neurones * 3) / 4)
            leyers.append(layer)

        leyers.append(generate_layer(input_size, 10, "softmax", "l1", 0.2))
        arch[f'W_{neurones_l1}_D_{depht}'] = leyers

log = {}
for arch_k in arch.keys():
    net = MyDNN(arch[arch_k], "cross-entropy", 5e-5)
    log[arch_k] = net.fit(training_samples, training_classifications, 200, 1024, 0.2, validation_samples,
                          validation_classifications)
pickle.dump(log, open(f'..\\LOG\\\\Log_first_arch.p', 'wb'))

log = pickle.load(open(f'..\\LOG\\\\Log_first_arch_after.p', 'rb'))
print('Test 1 layer relu and 1 softmax L1')
for key in log.keys():
    t = key.split('_')
    tirlt = f'Depth:{t[3]} and width{t[1]} '
    plot_graphs(log[key], tirlt)

for key in log.keys():
    print(key, f'Time : {log[key][199]["time"]} acc: {log[key][199]["acc"]}')
