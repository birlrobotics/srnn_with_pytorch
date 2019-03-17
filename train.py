from __future__ import print_function
import argparse
import torch
import numpy as np
import os
import sys
import cPickle
from torch import nn, optim
from model import SRNN
import matplotlib.pyplot as plt

###########################################################################################
#                                 SET SOME PARAMETERS                                     #
###########################################################################################

parser = argparse.ArgumentParser(description = "SRNN for human activities && \
                                                object affordance anticipation with Pytoch")

parser.add_argument('--data', type=str, default='./dataset',
                    help='location of the dataset')

parser.add_argument('--epoch', type=int, default=300,
                    help='number of epochs to train')

parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')

parser.add_argument('--clip', type=int, default=5,
                    help='gradient clipping') 
                    
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout parameter')

parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')  

parser.add_argument('--print_every', type=int, default=10,
                    help='number of steps for printing training and validation loss') 

parser.add_argument('--save_every', type=int, default=10,
                    help='number of steps for saving the model parameters') 

parser.add_argument('--save', type=str, default='./checkpoints_anticipation',
                    help='path to save the checkpoints')      

parser.add_argument('--pretrained', type=str, default='',
                    help='location of the pretrained model file')   

parser.add_argument('--architecture', type=str, default='detection',
                    help='architecture style: detection or anticipation')              

args = parser.parse_args()         

###########################################################################################
#                                   LOAD THE DATAS                                        #
###########################################################################################

path_to_dataset = args.data

validation_data = cPickle.load(open('{}/validation_data.pik'.format(path_to_dataset)))
Y_te_human = validation_data['labels_human']  
Y_te_human_anticipation = validation_data['labels_human_anticipation']    
X_te_human_disjoint = validation_data['features_human_disjoint']     
X_te_human_shared = validation_data['features_human_shared']      

Y_te_objects = validation_data['labels_objects']     # 25X212
Y_te_objects_anticipation = validation_data['labels_objects_anticipation']   #25X212
X_te_objects_disjoint = validation_data['features_objects_disjoint']     #(25, 212, 620)
X_te_objects_shared = validation_data['features_objects_shared']     #(25, 212, 400)

train_data = cPickle.load(open('{}/train_data.pik'.format(path_to_dataset)))
Y_tr_human = train_data['labels_human']     # 25X100
Y_tr_human_anticipation = train_data['labels_human_anticipation']  # 25X100
X_tr_human_disjoint = train_data['features_human_disjoint']    # 25X100X790
X_tr_human_shared = train_data['features_human_shared']     #25X100X400

Y_tr_objects = train_data['labels_objects']     # 25X212
Y_tr_objects_anticipation = train_data['labels_objects_anticipation']   #25X212
X_tr_objects_disjoint = train_data['features_objects_disjoint']     #(25, 212, 620)
X_tr_objects_shared = train_data['features_objects_shared']     #(25, 212, 400)

seq_length = Y_tr_human.shape[0]
num_te_data = len(Y_te_human)
num_sub_activities = int(np.max(Y_tr_human) - np.min(Y_tr_human) + 1)
num_affordances = int(np.max(Y_tr_objects) - np.min(Y_tr_objects) + 1)
num_sub_activities_anticipation = int(np.max(Y_tr_human_anticipation) - np.min(Y_tr_human_anticipation) + 1)
num_affordances_anticipation = int(np.max(Y_tr_objects_anticipation) - np.min(Y_tr_objects_anticipation) + 1)
inputJointFeatures = X_tr_human_shared.shape[2]
inputHumanFeatures = X_tr_human_disjoint.shape[2]
inputObjectFeatures = X_tr_objects_disjoint.shape[2]
assert(inputJointFeatures == X_tr_objects_shared.shape[2])

print ('#the length of sequence', seq_length) 
print ('#human sub-activities ',num_sub_activities)
print ('#object affordances ',num_affordances)
print ('#human sub-activities-anticipation ',num_sub_activities_anticipation)
print ('#object affordances_anticipation ',num_affordances_anticipation)
print ('shared features dim ',inputJointFeatures)
print ('human features dim ',inputHumanFeatures)
print ('object features dim ',inputObjectFeatures)

###########################################################################################
#                                   CREATE BATCHES                                        #
###########################################################################################

def get_batch(dataset, batch_size, n_data, device):
    global dataset_batch
    dataset_batch = []
    n_batch = n_data // batch_size
    for n in range(n_batch):
        for data in dataset:
            if len(data.shape) > 2:
                date_batch = data[:, n:n+batch_size, :]
            else:
                date_batch = data[:, n:n+batch_size]
            data_batch = torch.from_numpy(data_batch).to(device)
            dataset_batch.append(data_batch)
        yield dataset_batch

###########################################################################################
#                                   CALCULATE ACCURACY                                    #
###########################################################################################

def get_accuracy(log_output, lables, acc = True):
    # change to probability
    prob = torch.exp(log_output)
    top_p, top_class = prob.topk(1, dim=1)
    equals = top_class == lables.view(top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    if acc:
        return accuracy
    else:
        return equals, top_class

###########################################################################################
#                                     TRAIN MODEL                                         #
###########################################################################################

def run_network(net, dataset,validation_data, epoch = 100, batch_size = 10, lr = 0.001, clip = 5, print_every = 10, save_every= 10):

    # check the path to save checkpoints    
    if not os.path.exists(args.save):
	    os.makedirs(args.save)
    # set running device model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on {}".format(device))
    # initialize hidden states,and to device
    h_hiddens = net.init_hidden(Y_tr_human.shape[1], device)
    o_hiddens = net.init_hidden(Y_tr_objects.shape[1], device)
    val_h_hiddens = net.init_hidden(Y_te_human.shape[1], device)
    val_o_hiddens = net.init_hidden(Y_te_objects.shape[1], device)
    # put element of dataset and network into running device
    for i, data in enumerate(dataset):
        dataset[i] = torch.from_numpy(data).to(device)
    for i, data in enumerate(validation_data):
        validation_data[i] = torch.from_numpy(data).to(device)
    net.to(device)
    # define optimization method and criterion
    opt = optim.Adam(net.parameters(), lr = lr)
    criterion = nn.NLLLoss()
    # generate two lists to keep the loss
    running_loss, validation_loss = [], []
    for e in range(epoch):
        ###################################################################################
        #                         TRAINING ON TRAIN DATASET                               #
        ###################################################################################
        '''
            Because of the different number between human and object in all dataset,
            and the structure of the neural network I design just combine both the network 
            for h_a and the network for o_a, so all samples in the dataset will be put into
            the model instead of split them into batch size. 
        '''
        net.train()
        # for x_h_s, x_h_d, y_h, x_o_s, x_o_d, y_o in get_batch(dataset, batch_size, num_tr_data, device):
        x_h_s, x_h_d, y_h, x_o_s, x_o_d, y_o = dataset
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h_hidden = tuple([(each[0].data, each[1].data) for each in h_hiddens])
        o_hidden = tuple([(each[0].data, each[1].data) for each in o_hiddens])
        # zero accumulated gradients
        net.zero_grad()
        # get the output from the model
        outputs, h = net([x_h_s, x_h_d], [x_o_s, x_o_d], h_hidden, o_hidden)
        # calculate the loss and perform backprop
        h_loss = criterion(outputs[0], y_h.view(np.multiply(y_h.shape[0], y_h.shape[1])))
        o_loss = criterion(outputs[1], y_o.view(np.multiply(y_o.shape[0], y_o.shape[1])))
        train_loss = h_loss + o_loss
        train_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()

        ###################################################################################
        #                         TESTING ON VALIDATION DATASET                           #
        ###################################################################################
        # turn off the gradients for vaildation, save memory and computations
        with torch.no_grad():
            # set running model as vaildation
            net.eval()
            # run the model with the validation dataset
            x_h_s, x_h_d, y_h, x_o_s, x_o_d, y_o = validation_data
            h_hidden = tuple([(each[0].data, each[1].data) for each in val_h_hiddens])
            o_hidden = tuple([(each[0].data, each[1].data) for each in val_o_hiddens])   
            outputs, h = net([x_h_s, x_h_d], [x_o_s, x_o_d], h_hidden, o_hidden)
            h_loss = criterion(outputs[0], y_h.view(np.multiply(y_h.shape[0], y_h.shape[1])))
            o_loss = criterion(outputs[1], y_o.view(np.multiply(y_o.shape[0], y_o.shape[1])))
            val_loss = h_loss + o_loss
            # calcalate the accuracy
            h_accuracy = get_accuracy(outputs[0], y_h)
            o_accuracy = get_accuracy(outputs[1], y_o)
            # get the result at the last epoch, just for debug
            if e == (epoch - 1):
                h_pre_equals, h_pre_class = get_accuracy(outputs[0], y_h, False)
                o_pre_equals, o_pre_class = get_accuracy(outputs[1], y_o, False)
        # save the loss
        running_loss.append(train_loss.item())
        validation_loss.append(val_loss.item())
        if e % print_every == 9:
            print("{0}/{1}, train_loss:{2},  val_loss:{3}, h_accuracy:{4}, o_accuracy:{5}".format \
                           (e, epoch, train_loss.item(), val_loss.item(), h_accuracy, o_accuracy))
        # save the model
        if e == (epoch-1) or e % save_every == (save_every-1):
            model_name = "{0}_{1}_epochs.net".format(args.architecture, e+1)
            checkpoints = {'h_n_labels': net.h_n_labels,
                           'o_n_labels': net.o_n_lables,
                           'lstm1_input_size': net.lstm1_input_size,
                           'lstm2_input_segment_size': net.lstm2_input_segment_size,
                           'n_hidden1': net.n_hidden1,
                           'n_hidden2': net.n_hidden2,
                           'state_dict': net.state_dict()}
            with open(os.path.join(args.save, model_name), "wb") as f:
                torch.save(checkpoints, f)

    # draw the losses
    plt.plot(running_loss, label="Training loss")
    plt.plot(validation_loss, label="Validation loss")
    plt.legend(frameon=False)
    plt.show()

###########################################################################################
#                                     TRAIN MODEL                                         #
###########################################################################################

if not args.pretrained:
    if args.architecture == "detection":
        # generate the model
        model = SRNN(num_sub_activities, num_affordances, 
                     inputJointFeatures, np.array([inputHumanFeatures, inputObjectFeatures]), 
                     drop_prob= args.dropout)
        
        # print the all the number of the model parameters
        print("number of model parameters: ", sum(param.numel() for param in model.parameters()))
        # generate a specified dataset
        train_set = [X_tr_human_shared, X_tr_human_disjoint, Y_tr_human, 
                   X_tr_objects_shared, X_tr_objects_disjoint, Y_tr_objects]
        validation_set = [X_te_human_shared, X_te_human_disjoint, Y_te_human, 
                          X_te_objects_shared, X_te_objects_disjoint, Y_te_objects]
        # training
        run_network(model,  train_set, validation_set, epoch = args.epoch, batch_size = args.batch_size, lr = args.lr,
                    clip = args.clip, print_every = args.print_every, save_every= args.save_every)
        
        
