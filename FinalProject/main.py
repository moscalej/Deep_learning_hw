
from test.evaluate import evaluate
#%%

T2_ADR = r"C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\check_point\vgg16_t2\09-0.7635.h5"
T4_ADR = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\check_point\vgg16_t2\09-0.7635.h5'
T5_ADR = r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\check_point\vgg16_t2\09-0.7635.h5'

model_paths = {
    2: T2_ADR,
    4: T4_ADR,
    5: T5_ADR
}

image_path = r"C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\FinalProject\data\test"

im = evaluate(image_path, model_paths)
print(im)

