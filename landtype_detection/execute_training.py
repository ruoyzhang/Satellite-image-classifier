from train import train
from custom_dset_new import train_val_test_split, custom_dset
from pretrained_inceptionv3 import pretrained_inception_v3


# setting up variables
data_dir = '../../data/images/'
save_dir = '../../data/CNN_model_landtype/'
num_class = 6
bs = 6
num_epoch = 8
lr = 3e-4



# run
loss_record, acc_record, model, test_data = train(data_dir = data_dir, save_dir = save_dir, num_class = 6, num_epoch = num_epoch, bs = bs, use_cuda = True)