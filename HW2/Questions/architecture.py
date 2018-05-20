from mydnn import *
import pickle
import gzip
import urllib.request
from sklearn.preprocessing import scale
from mydnn import MyDNN
from Macros import generate_layer

arch = {}
log_path = '.\\LOG'
data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
training_samples = scale(train_set[0], axis=0, with_std=False)
validation_samples = scale(valid_set[0], axis=0, with_std=False)
training_classifications = train_set[1]
validation_classifications = valid_set[1]
num_samples, num_pixels = training_samples.shape

arch = {}
for depht in [1, 2, 3]:
    for neurones_l1 in [128, 254, 512]:
        input_size = num_pixels
        neurones = neurones_l1
        leyers = []
        for index in range(depht):
            layer = generate_layer(input_size, neurones, 'relu', "l2", 0.2)
            input_size = neurones
            neurones = neurones / 2
            leyers.append(layer)

        leyers.append(generate_layer(input_size, 10, "softmax", "l1", 0.2))
        arch[f'W_{neurones_l1}_D_{depht}'] = leyers

log = {}
for arch_k in arch.keys():
    net = MyDNN(arch[arch_k], "cross-entropy", 5e-5)
    log[arch_k] = net.fit(
        training_samples, training_classifications, 100, 1024, 0.2,
        validation_samples, validation_classifications)
pickle.dump(log, open(f'..\\LOG\\\\Log_first_arch.p', 'wb'))
