from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
datagen.fit(x_train)
tbCallBack = keras.callbacks.TensorBoard(log_dir='.Code/logs/Incep3_ras_dg_run5/', histogram_freq=0, batch_size=32, write_graph=True,
                                         write_grads=True, write_images=True, embeddings_freq=0,
                                         embeddings_layer_names=None, embeddings_metadata=None)

history_fully = model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=3000),
                                epochs=100,
                                validation_data=(x_test, y_test),callbacks=[tbCallBack])
