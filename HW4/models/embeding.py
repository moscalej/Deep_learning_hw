"""
# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation All Rights Reserved.
# The source code contained or described herein and all documents related
# to the source code ("Material") are owned by Intel Corporation or its
# suppliers or licensors. Title to the Material remains with Intel Corp-
# oration or its suppliers and licensors. The Material contains trade
# secrets and proprietary and confidential information of Intel Corpor-
# ation or its suppliers and licensors. The Material is protected by world-
# wide copyright and trade secret laws and treaty provisions. No part of
# the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
# No license under any patent, copyright, trade secret or other intellect-
# ual property right is granted to or conferred upon you by disclosure or
# delivery of the Materials, either expressly, by implication, inducement,
# estoppel or otherwise. Any license under such intellectual property
# rights must be express and approved by Intel in writing.
#
#           #           #           #
# Authors:  Alejandro Moscoso <alejandro.moscoso@intel.com>
#
"""
import numpy as np
from models.Lenguage import load_imbd, data_generator_embeding
import keras.layers as kl
import keras.models as km
import keras

def embeding_network():
    standardModel = km.Sequential()
    standardModel.add(kl.Embedding(input_dim=1,output_dim=128))
    standardModel.add(kl.Dense(input_dim=128, output_dim=128 ))
    standardModel.add(kl.Dense(10_000, activation='softmax'))
    standardModel.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    return standardModel
Data, Labels, word_to_id, id_to_word = load_imbd(10_000, 200)
model = embeding_network()
model.fit_generator(data_generator_embeding(Data,batch_size=16,voc_size=10_000),50_000//1024,10)