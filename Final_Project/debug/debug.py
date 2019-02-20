

from Final_Project.models.Algorithm_sketch import Puzzle, plot_crops, predict_2
from Final_Project.models.Preprocesses import pre_process_data, processed2train_2_chanel, processed2train

SHAPE = 64

path_train = [r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\toy_data']

processed_train = pre_process_data(path_train, shape=SHAPE, cuts=4,normalize=False)
t = processed_train[24]
t_1 = processed_train[22]
t_2 = processed_train[21]
t_3 = processed_train[16]
t_4 = processed_train[15]

t = [x[0] for x in t]
t.extend([t_1[5][0],t_2[7][0],t_3[10][0],t_4[9][0]])
print('keelo')
plot_crops(t)


predict_2(t)