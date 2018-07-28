"""
Authors :       Zachary Bamberger
                Alejandro Moscoso
"""
import os
import math
from models.Data_set_creator import DSC


# import keras
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

if __name__ == "__main__":

    # Determine the right path to the images.
    # TODO: make this an input to the script later on
    if "Zach" in os.environ.get('USERNAME'):
        image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\images"
        documents_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\documents"
    else:
        image_path = r"D:\Ale\Documents\Technion\Deep Learning\DL_HW\FinalProject\data\images"

    t_2_dataset = DSC(images_path=image_path, t_value=2)
    t_4_dataset = DSC(images_path=image_path, t_value=4)
    t_5_dataset = DSC(images_path=image_path, t_value=5)









