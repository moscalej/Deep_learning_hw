from random import randint

from Final_Project.models.Algorithm_sketch import Puzzle, plot_crops, predict_2, predict_3
from Final_Project.models.Preprocesses import pre_process_data, processed2train_2_chanel, processed2train

SHAPE = 64

# path_train = [r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\toy_data']
path_train = [r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\images', r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\documents']
processed_train = pre_process_data(path_train, shape=SHAPE, cuts=5, normalize=False)


def create_test(images: list, cuts: int):
    number_pic = len(images)
    samples = []
    for index, picture in enumerate(images):
        crops = [crop[0] for crop in picture]
        # for i in range(cuts - 1):
        #     ood = randint(0, number_pic - 1)
        #     if ood != index:
        #         ood_pic = images[ood][randint(0, cuts ** 2 - 1)][0]
        #         crops.append(ood_pic)
        samples.append(crops)
    return samples


p = create_test(processed_train, 5)

print(f"accuracy: {predict_3(p)}")
