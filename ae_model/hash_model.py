from keras.models import Model,load_model
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import numpy as np
import scipy.io as sio

from ae_model.preprocess import load_data

from ae_model.conv_model import HashSupervisedAutoEncoderModel



def scheduler(epoch):
    if epoch <= 30:
        return 0.1
    if epoch <= 55:
        return 0.02
    if epoch <= 75:
        return 0.004
    return 0.0008



#which_data = "mnist"
which_data = "1d_mnist"
num_classes = 10
stack_num = 1   # number of stack in the ResNet network
batch_size = 64   # number of training batch per step
epochs = 1       # number of training epoch

# weight in the loss function
alpha = 1e-1    # weight of binary loss term
beta = 1e-1     # weight of evenly distributed term
gamma = 1   # weight of recovery loss term

hash_bits = 5  # length of hash bits to encode

test_size = 10

log_path='/home/toliks/PycharmProjects/dmbi-imperva/ae_model/logs/'
save_path='/home/toliks/PycharmProjects/dmbi-imperva/ae_model/cp_models/'
hash_file_path = "./"+which_data+"_hash_32bits_res_testset_su_ae.mat"



(x_train, y_train), (x_test, y_test) = load_data(which_data)
(_, img_rows, img_cols, img_channels) = x_train.shape
(x_train, y_train)=(x_train[:10000,:,:], y_train[:10000,:])
# build our Supervised Auto-encoder Hashing network
hash_su_ae_model = HashSupervisedAutoEncoderModel(img_rows, img_cols, img_channels, num_classes, stack_num,
                                                             hash_bits, alpha, beta, gamma)

resnet = Model(inputs=hash_su_ae_model.img_input, outputs=[hash_su_ae_model.y_predict, hash_su_ae_model.y_decoded])


resnet.summary()

sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
resnet.compile(optimizer=sgd,loss={'y_predict': 'categorical_crossentropy','y_decoded': hash_su_ae_model.net_loss}, metrics={'y_predict': 'accuracy'},loss_weights=[1, 1.])

tb_cb = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=False)
change_lr = LearningRateScheduler(scheduler)
chk_pt = ModelCheckpoint(filepath=save_path + 'chk_pt.h5', monitor='val_y_predict_acc', save_best_only=True, mode='max',
                         period=1)
cbks = [change_lr, tb_cb, chk_pt]

resnet.fit(x_train, {"y_predict": y_train, "y_decoded": x_train}, epochs=epochs, batch_size=batch_size, callbacks=cbks,validation_data=(x_test, [y_test, x_test]))



#resnet = load_model(save_path+'chk_pt.h5', compile=False)
#resnet = load_model('/Users/toliks/PycharmProjects/DataHack/bits32/chk_pt.h5', compile=False)

resnet.summary()
print("hash code length is:", resnet.get_layer("hash_x").output.get_shape().as_list()[-1])
hash_model = Model(input=resnet.input,output=resnet.get_layer("hash_x").output)



batches = len(x_test) / test_size

print("generate hash now ...")
for i in range(int(batches)):
    x_batch = x_test[i * test_size:(i + 1) * test_size, :, :, :]
    y_batch = y_test[i * test_size:(i + 1) * test_size]
    hash_temp = hash_model.predict(x_batch)

    hash_temp = np.array(hash_temp, float)
    hash_temp = np.reshape(hash_temp, [test_size, hash_bits])

    if i == 0:
        hash_output = hash_temp
    else:
        hash_output = np.concatenate((hash_output, hash_temp))
print(hash_output.shape)
hash_binary = np.where(hash_output > 0.5, 1, 0)

print("save hash to file ..." + hash_file_path)
sio.savemat(hash_file_path, {"hash_org": hash_output, "hash_bin": hash_binary, "label": y_test})
