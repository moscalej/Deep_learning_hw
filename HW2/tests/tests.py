from Nodes import *
from Layer import *
from mydnn import mydnn
from Macros import *

class NodesTest(unittest.TestCase):

    def test_mse(self):

        x = np.array([[1, 1, 4],
                      [1, 2, 2],
                      [-1, 3, 5],
                      [6, 7, -5]]) * 0.1
        w = np.array([[-0.24701764, -0.06520847, 0.27938292, -0.30231493],
                      [0.36299324, 0.48340068, -0.33615776, 0.09733394]])
        b = np.zeros([2, 1])
        labels = np.array([[4, 2, 6],
                           [2, 0.1, 2]]) * 0.1
        m = Multiplication()
        add = Add_node()
        sig = Sigmoid()
        soft = SoftMax()
        relu = Relu()
        mse = MSE()
        ent = Entropy()
        none = NoneNode()
        for iter in range(1000):
            after_mult = m.forward(x, w)
            after_add = add.forward(after_mult, b)
            total = none.forward(after_add)
            mse.forward(after_add, labels, 2)

            out = none.backward(mse.gradient)
            b_d, a = add.backward(out)
            xm, wm = m.backward(b_d)
            w = w - 0.5 * wm
            b = b - 0.5 * np.sum(b_d, axis=1).reshape(b.shape)

        print(labels)
        print(total)
        out = np.sum(np.sum(labels - total))
        self.assertLessEqual(out, 0.01, msg=f'MSE fail : {out}')

    def test_softmax(self):
        m = Multiplication()
        add = Add_node()
        sig = Sigmoid()
        soft = SoftMax()
        relu = Relu()
        mse = MSE()
        ent = Entropy()
        none = NoneNode()
        x = labels = np.array([[0.1, 1],
                               [1, 2],
                               [-1, 3],
                               [6, 7]]) * 0.1
        w = np.array([[-0.24701764, -0.06520847, 0.27938292, -0.30231493],
                      [0.36299324, 0.48340068, -0.33615776, 0.09733394]])
        b = np.zeros([2, 1])
        labels = np.array([[0, 1],
                           [1, 0]])

        for iter in range(1000):
            after_mult = m.forward(x, w)
            after_add = add.forward(after_mult, b)
            total = soft.forward(after_add)

            ent.forward(total, labels, 2)
            out = soft.backward(ent.gradient)
            b_d, a = add.backward(out)
            xm, wm = m.backward(b_d)
            w = w - 0.5 * wm
            b = b - 0.5 * np.sum(b_d, axis=1).reshape(b.shape)

        print(labels)
        print(total)
        self.assertEqual(np.argmax(labels), np.argmax(total))


class LayerTest(unittest.TestCase):

    def test_layer1(self):
        for non_linear in ["relu", "sigmoid", "softmax", "none"]:
            for loss in ['l1', 'l2']:
                layer1 = Layer(9, 3, non_linear, loss, 0.2, 0.1)
                layer_input = np.ones([9, 8])
                forward_1 = layer1.forward(layer_input)
                backward_1_bias = layer1.backward(forward_1)
                print('Pass')
        self.assertTrue(True)

    def test_layer_2_mse(self):
        x = np.array([[1, 1, 4],
                      [1, 2, 2],
                      [-1, 3, 5],
                      [6, 7, -5]]) * 0.1
        b = np.zeros([2, 1])
        labels = np.array([[4, 2, 6],
                           [2, 0.1, 2]]) * 0.1
        mse = MSE()

        layer1 = Layer(4, 2, 'none', 'l2', 0.2, 1e-9)
        for iter in range(1000):
            total = layer1.forward(x)
            mse.forward(total, labels, 3)
            layer1.backward(mse.gradient)
        print('')
        print(labels)
        print(total)
        out = np.sum(np.sum(labels - total))
        self.assertLessEqual(out, 0.001, msg=f'MSE fail : {out}')

    def test_layer2_softmax_l2(self):
        x = np.array([[1, 1, 4],
                      [1, 2, 2],
                      [-1, 3, 5],
                      [6, 7, -5]]) * 0.1
        b = np.zeros([2, 1])
        labels = np.array([[0, 1, 1],
                           [1, 0, 1]])
        ent = Entropy()
        layer1 = Layer(4, 2, 'softmax', 'l2', 0.5, 1e-9)
        for iter in range(1000):
            total = layer1.forward(x)
            ent.forward(total, labels, 3)
            layer1.backward(ent.gradient)
        print('')
        print(labels)
        print(total)
        self.assertEqual(np.argmax(labels), np.argmax(total))

    def test_layer2_softmax_l1(self):
        x = np.array([[1, 1, 4],
                      [1, 2, 2],
                      [-1, 3, 5],
                      [6, 7, -5]]) * 0.1
        b = np.zeros([2, 1])
        labels = np.array([[0, 1, 1],
                           [1, 0, 1]])
        ent = Entropy()
        layer1 = Layer(4, 2, 'softmax', 'l1', 0.5, 1e-9)
        for iter in range(1000):
            total = layer1.forward(x)
            ent.forward(total, labels, 3)
            layer1.backward(ent.gradient)
        print('')
        print(labels)
        print(total)
        self.assertEqual(np.argmax(labels), np.argmax(total))


class Dnn_test(unittest.TestCase):
    def test_regresion(self):
        x = np.array([[1, 1, 4],
                      [1, 2, 2],
                      [-1, 3, 5],
                      [6, 7, -5]]) * 0.1
        labels = np.array([[4, 2, 6],
                           [2, 0.1, 2]]) * 0.1

        big_layers = [generate_layer(4, 100, "none", "l1"),
                      generate_layer(100, 2, "none", "l1"),
                      ]
        big_net = mydnn(big_layers, "MSE", 1e-9, verbose=False)
        big_net.fit(x.T, labels.T, 1000, 1, 0.2, x.T, labels.T)
        total = big_net.predict(x.T)
        print(labels)
        print(total)
        out = np.sum(np.sum(labels - total))
        self.assertLessEqual(out, 0.001, msg=f'MSE fail : {out}')

    def test_regresion(self):
        x = np.array([[1, 1, 4],
                      [1, 2, 2],
                      [-1, 3, 5],
                      [6, 7, -5]]) * 0.1
        labels = np.array([[4, 2, 6],
                           [2, 0.1, 2]]) * 0.1

        big_layers = [generate_layer(4, 100, "none", "l2"),
                      generate_layer(100, 2, "none", "l2"),
                      ]
        big_net = mydnn(big_layers, "MSE", 1e-5)
        big_net.fit(x.T, labels.T, 1000, 1, 0.2, x.T, labels.T, verbose=False)
        total = big_net.predict(x.T)
        print(labels)
        print(total)
        out = np.sum(np.sum(labels - total))
        self.assertLessEqual(out, 0.001, msg=f'MSE fail : {out}')

    def test_regresion(self):
        x = np.array([[1, 1, 4],
                      [1, 2, 2],
                      [-1, 3, 5],
                      [6, 7, -5]]) * 0.1
        labels = np.array([[4, 2, 6],
                           [2, 0.1, 2]]) * 0.1

        big_layers = [generate_layer(4, 100, "relu", "l2"),
                      generate_layer(100, 2, "none", "l2"),
                      ]
        big_net = mydnn(big_layers, "MSE", 1e-4)
        big_net.fit(x.T, labels.T, 1000, 1, 0.2, x.T, labels.T, verbose=False)
        total = big_net.predict(x.T)
        print('')
        print(labels)
        print(total)
        out = np.sum(np.sum(labels - total))
        self.assertLessEqual(out, 0.001, msg=f'MSE fail : {out}')

if __name__ == '__main__':
    unittest.main()
