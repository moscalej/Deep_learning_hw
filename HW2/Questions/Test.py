import pickle
import gzip
import urllib.request
from sklearn.preprocessing import scale
from mydnn import MyDNN
from Macros import generate_layer
from Macros import plot_graphs
import pickle
import os

log_path = '.\\LOG'
print(log_path)
# data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
# urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
training_samples = scale(train_set[0], axis=0, with_std=False)
validation_samples = scale(valid_set[0], axis=0, with_std=False)
training_classifications = train_set[1]
validation_classifications = valid_set[1]
num_samples, num_pixels = training_samples.shape

# First Experiment
layers = [generate_layer(num_pixels, 128, "relu", "l2", 0.2)]
layers.append(generate_layer(128, 10, "softmax", "l2", 0.2))
log = {}
for batch in [128, 1024, 6e5]:
    net = MyDNN(layers, "cross-entropy")
    log[batch] = net.fit(
        training_samples, training_classifications, 100, batch, 0.4,
        validation_samples, validation_classifications)
pickle.dump(log, open(f'{log_path}\\Log_first.p', 'wb'))

log = pickle.load(open(f'{log_path}\\Log_first.p', 'rb'))

print('Test 1 layer relu and 1 softmax')
for batch_size in log.keys():
    print(f'Batch size: {batch_size}')
    plot_graphs(log[batch_size])

# Second Experiment L1
import numpy as np

layers = [generate_layer(num_pixels, 128, "relu", "l1", 0.2)]
layers.append(generate_layer(128, 10, "softmax", "l1", 0.2))
log = {}
for weight in [5e-5, 5e-3, 5e-2]:
    net = MyDNN(layers, "cross-entropy", weight)
    log[-round(np.log10(weight))] = net.fit(
        training_samples, training_classifications, 200, 1024, 0.4,
        validation_samples, validation_classifications)
pickle.dump(log, open(f'{log_path}\\Log_first_l1.p', 'wb'))

log = pickle.load(open(f'{log_path}\\Log_first_l1.p', 'rb'))
print('Test 1 layer relu and 1 softmax L1')
for batch_size in log.keys():
    print(f'Lambda size: {batch_size}')
    plot_graphs(log[batch_size])

# Second Experiment L2
import numpy as np

layers = [generate_layer(num_pixels, 128, "relu", "l2", 0.2)]
layers.append(generate_layer(128, 10, "softmax", "l2", 0.2))
log = {}
for weight in [5e-5, 5e-4, 5e-3]:
    net = MyDNN(layers, "cross-entropy", weight)
    log[-round(np.log10(weight))] = net.fit(
        training_samples, training_classifications, 100, 1024, 0.2,
        validation_samples, validation_classifications)
pickle.dump(log, open(f'{log_path}\\Log_first_l2.p', 'wb'))

log = pickle.load(open(f'{log_path}\\Log_first_l2.p', 'rb'))
from Macros import plot_graphs

print('Test 1 layer relu and 1 softmax L2')
for batch_size in log.keys():
    print(f'Lambda size: {batch_size}')
    plot_graphs(log[batch_size])
