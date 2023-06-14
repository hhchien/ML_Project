from distutils.command.config import config
import numpy as np
import os
import sys
import time
from sklearn import metrics
import pickle
import torch
from sklearn.model_selection import train_test_split
from model.model import NeuSomaticNet
import torch.optim as optim
import torch.nn as nn
from model.dataloader import CustomDataset
import wandb

from config import config_param


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
## CUDA for PyTorch
torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()

window_size = int(sys.argv[1])
learning_rate = float(sys.argv[2])

# for wandb
os.environ["WANDB_API_KEY"] = "3607c36d6772a9ce6c02ca86209013650366fc23"
wandb.login()
# init wandb
wandb.init(project="Mutation_Detection_Comparation_Test", name='All_lab_ws_{}_with_learning_rate_{}'.format(window_size, learning_rate))

def load_pickle(file):
    with open(file, 'rb') as f:
        matrix = pickle.load(f)
    return matrix


def split_data(input_matrix, labels):
    input_matrix_train, input_matrix_val, labels_train, labels_val = train_test_split(input_matrix, labels, test_size=0.1, random_state=42)
    return (input_matrix_train, labels_train), (input_matrix_val, labels_val)


def make_weights_for_balanced_classes(count_class, nclasses):
        w_t = [0] * nclasses
        count_class = list(count_class)
        N = float(sum(count_class))
        
        for i in range(nclasses):
            w_t[i] = (1 - (float(count_class[i]) / float(N))) / float(nclasses)
        w_t = np.array(w_t)

        return w_t

print("Training with WES datasets --- Window size = {}".format(window_size))

list_chr = []
num_chr = np.arange(22)
for i in num_chr:
    list_chr.append('chr{}'.format(i+1))
print(num_chr)

matrix_train = []
label_train = []
matrix_test = []
label_test = []

list_lab = ['WES_EA_T_1','WES_FD_T_3', 'WES_IL_T_2', 'WES_LL_T_2', 'WES_NC_T_3', 'WES_NV_T_1']
for lab_name in list_lab:
    save_pickle_dir_train = './data/pickle/{}_ws{}_pickle'.format(lab_name, window_size)
    save_pickle_dir_test = './data_test/pickle/{}_ws{}_pickle'.format(lab_name, window_size) 
    # get input matrix and label for training model 
    for chr in list_chr:
        chr_dir_train = os.path.join(save_pickle_dir_train, chr)
        matrix_pickle_train = os.path.join(chr_dir_train, 'input_matrix.pickle')
        label_pickle_train = os.path.join(chr_dir_train, 'output_label.pickle')
        matrix_train.extend(load_pickle(matrix_pickle_train))
        label_train.extend(load_pickle(label_pickle_train))
        
        chr_dir_test = os.path.join(save_pickle_dir_test, chr)
        matrix_pickle_test = os.path.join(chr_dir_test, 'input_matrix.pickle')
        label_pickle_test = os.path.join(chr_dir_test, 'output_label.pickle')
        matrix_test.extend(load_pickle(matrix_pickle_test))
        label_test.extend(load_pickle(label_pickle_test))


matrix_train = np.array(matrix_train)
label_train = np.array(label_train)
matrix_test = np.array(matrix_test)
label_test = np.array(label_test)

# normalize matrix
for i in range(len(matrix_train)):
    if np.max(matrix_train[i, :, :, 1]) !=0:
        matrix_train[i, :, :, 1] /= np.max(matrix_train[i, :, :, 1])
    if np.max(matrix_train[i, :, :, 2]) !=0:
        matrix_train[i, :, :, 2] /= np.max(matrix_train[i, :, :, 2])

for i in range(len(matrix_test)):
    if np.max(matrix_test[i, :, :, 1]) !=0:
        matrix_test[i, :, :, 1] /= np.max(matrix_test[i, :, :, 1])
    if np.max(matrix_test[i, :, :, 2]) !=0:
        matrix_test[i, :, :, 2] /= np.max(matrix_test[i, :, :, 2])

print(len(matrix_train), len(label_train))
# split data to training set, validation set and test set
data_train, data_val = split_data(matrix_train, label_train)
input_matrix_train, labels_train = data_train
input_matrix_val, labels_val = data_val

input_matrix_test = np.array(matrix_test)
labels_test = np.array(label_test)

print(input_matrix_train.shape, labels_train.shape)
print(input_matrix_val.shape, labels_val.shape)
print(input_matrix_test.shape, labels_test.shape)

# For training set
num_non_somatic_train = np.sum(labels_train[:, 0]) 
num_snv_train = np.sum(labels_train[:, 1]) 
num_indel_train = np.sum(labels_train[:, 2]) 

print("Training set: Non-somatic: {}, SNV: {}, INDEL: {}".format(num_non_somatic_train, num_snv_train, num_indel_train))

# For validation set
num_non_somatic_val = np.sum(labels_val[:, 0]) 
num_snv_val = np.sum(labels_val[:, 1]) 
num_indel_val = np.sum(labels_val[:, 2]) 

print("Validation set: Non-somatic: {}, SNV: {}, INDEL: {}".format(num_non_somatic_val, num_snv_val, num_indel_val))

num_data_train = len(input_matrix_train)
num_data_val = len(input_matrix_val)
num_data_test = len(input_matrix_test)

# compute balance weight
#count the number of each classes
count_class = [0] * 3
for i in range(len(labels_train)):
    count_class[np.argmax(labels_train[i])] += 1
weight_balance = make_weights_for_balanced_classes(count_class, len(labels_train[0]))    
print("Balanced weights:", weight_balance)
weight_balance = torch.FloatTensor(weight_balance)
weight_balance = weight_balance.to(device)
    
# convert to tensor torch (Chanels, Heigh, Width)
input_matrix_train =  torch.permute(torch.from_numpy(input_matrix_train), (0, 3, 1, 2))
input_matrix_val =  torch.permute(torch.from_numpy(input_matrix_val), (0, 3, 1, 2))
input_matrix_test =  torch.permute(torch.from_numpy(input_matrix_test), (0, 3, 1, 2))

## Create Dataloader 
params = {'batch_size': 128,
        'shuffle': True,
        'num_workers': 3}
print("Config:", params)

training_set = CustomDataset(input_matrix_train, labels_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = CustomDataset(input_matrix_val, labels_val)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

test_set = CustomDataset(input_matrix_test, labels_test)
test_generator = torch.utils.data.DataLoader(test_set, **params)


####################################
## Create Model 
model = NeuSomaticNet(num_channels=3, dim=4, window_size=window_size)
model.to(device)
# print(model)

## Cross Entropy Loss 
error = nn.CrossEntropyLoss(weight_balance)

## Optimizer
# learning_rate = 0.0001
momentum = 0.9
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lmbda = lambda epoch: 0.9 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)


####################################
# Training
max_epochs = 100
lr_reduce_proportion = 0.3
lr_reduce_epoch = int(max_epochs * lr_reduce_proportion)
print(f"Training infor:\n - epoch: {max_epochs} \n - Leanring rate: {learning_rate} \n - Learning rate scheduler: True \n - Early stopping: True \n")

# measure how long training is going to take
print("[INFO] training the network...")

# save loss to plot
train_losses = []
valid_losses = []
f1_scores = []

# Loop over epochs
valid_loss_min = 100.0
f1_max = 0.0

# for early stopping
early_stop = 5
count_stop = 0

epsilon = 0.0002  
prev_loss = 0

for epoch in range(max_epochs):
    start_time = time.time()
    train_loss = 0.0
    valid_loss = 0.0
    train_acc = 0
    val_acc = 0
    
    model.train()
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        # local_labels = local_labels#.type(torch.LongTensor)   # casting to long
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, _ = model(local_batch)
        loss = error(outputs, local_labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item() * local_batch.size(0)
        train_acc += np.sum(outputs.data.cpu().numpy().argmax(-1) == local_labels.data.cpu().numpy().argmax(-1))
        
    
    # Validation

    model.eval()
    truth = []
    pred = []
    for local_batch, local_labels in validation_generator:
        # Transfer to GPU
        # local_labels = local_labels#.type(torch.LongTensor)   # casting to long
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        truth.append(local_labels.data.cpu())

        # forward + backward + optimize
        outputs, _ = model(local_batch)
        pred.append(outputs.data.cpu())
        loss = error(outputs, local_labels)
        
        # update-average-validation-loss 
        valid_loss += loss.item() * local_batch.size(0)
        val_acc += np.sum(outputs.data.cpu().numpy().argmax(-1) == local_labels.data.cpu().numpy().argmax(-1))
    
    truth = torch.cat(truth).numpy()
    pred = torch.cat(pred).numpy()

    pred_class = np.argmax(pred, axis=1)
    truth_class = np.argmax(truth, axis=1)
    f1_micro = metrics.f1_score(y_true=truth_class, y_pred=pred_class, average='weighted')
    f1_scores.append(f1_micro)
    train_losses.append(train_loss / num_data_train)
    valid_losses.append(valid_loss / num_data_val)

    if abs(valid_losses[-1] - prev_loss) <= epsilon:
        count_stop += 1
    else:
        count_stop = 0

    if count_stop == early_stop:
        print(f'Stopping training in epoch {epoch+1}')
        break
    
    prev_loss = valid_losses[-1]

    # get learning rate
    print(f'[epoch: {epoch + 1} -- Lr: {scheduler.get_lr()} -- time: {time.time()-start_time}s]  --- Acc_train: {train_acc / num_data_train} --- loss_train : {train_loss / num_data_train} --- Acc_val: {val_acc / num_data_val} --- loss_val : {valid_loss / num_data_val} --- f1_score: {f1_micro}')

    wandb.log({"acc_train": train_acc / num_data_train, "loss_train": train_loss / num_data_train, "acc_val": val_acc / num_data_val, "loss_val": valid_loss / num_data_val, "f1_score": f1_micro, "time":time.time()-start_time})
    
    # save model 
    torch.save(model, config_param.folder_save_model + 'last_classifier_type_somatic_model_full_wes_ws{}.pth'.format(window_size))

    if f1_micro > f1_max:
        f1_max = f1_micro
        torch.save(model, config_param.folder_save_model + 'best_classifier_type_somatic_model_full_wes_ws{}.pth'.format(window_size))

    if epoch >= lr_reduce_epoch:
        scheduler.step()

wandb.finish()
# =============================================================================================
# testing model 

def check(labels, predicts):
    predict_np = np.array(predicts)
    np.save('predict.npy', predict_np)
    count_true = np.zeros(3)
    count_total = np.zeros(3)

    for i in range(len(predicts)):
        count_total[labels[i]]+=1
        if labels[i] ==predicts[i]:
            count_true[labels[i]] += 1
    return count_true, count_total


# test with best model
print('****** Testing model with best checkpoint******')
model = torch.load(config_param.folder_save_model + 'best_classifier_type_somatic_model_full_wes_ws{}.pth'.format(window_size))
model.eval()
truth = []
predict = []

for local_batch, local_labels in test_generator:
    # Transfer to GPU
    local_labels = local_labels
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    # forward + backward + optimize
    outputs, _ = model(local_batch)

    truth.extend(local_labels.data.cpu().numpy().argmax(-1))
    predict.extend(outputs.data.cpu().numpy().argmax(-1))

count_true, count_total = check(truth, predict)
print(count_true, count_total, count_true/count_total, np.sum(count_true)/np.sum(count_total))

#  confusion matrix
print(metrics.classification_report(truth, predict, labels=[0, 1, 2]))

# test with last model
print('****** Testing model with last checkpoint******')
model = torch.load(config_param.folder_save_model + 'last_classifier_type_somatic_model_full_wes_ws{}.pth'.format(window_size))
model.eval()
truth = []
predict = []

for local_batch, local_labels in test_generator:
    # Transfer to GPU
    local_labels = local_labels
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    # forward + backward + optimize
    outputs, _ = model(local_batch)

    truth.extend(local_labels.data.cpu().numpy().argmax(-1))
    predict.extend(outputs.data.cpu().numpy().argmax(-1))

count_true, count_total = check(truth, predict)
print(count_true, count_total, count_true/count_total, np.sum(count_true)/np.sum(count_total))

#  confusion matrix
print(metrics.classification_report(truth, predict, labels=[0, 1, 2]))
    