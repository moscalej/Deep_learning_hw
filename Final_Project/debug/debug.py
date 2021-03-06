from random import randint

from Final_Project.models.Algorithm_sketch import Puzzle, plot_crops, predict_2, predict_3
from Final_Project.models.Preprocesses import pre_process_data, processed2train_2_chanel, processed2train

SHAPE = 64

# path_train = [r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\toy_data']
images_path = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\images'
documents_path = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\data\documents'
path_train = [images_path, documents_path]
processed_train = pre_process_data(path_train, shape=SHAPE, cuts=2, normalize=False)


def create_test(images: list, cuts: int):
    number_pic = len(images)
    samples = []
    for index, picture in enumerate(images):
        crops = [crop[0] for crop in picture]
        for i in range(cuts - 1):
            ood = randint(0, number_pic - 1)
            if ood != index:
                ood_pic = images[ood][randint(0, cuts ** 2 - 1)][0]
                crops.append(ood_pic)
        samples.append(crops)
    return samples


p = create_test(processed_train, 2)

predict_3(p)
