from __future__ import print_function
import torch
import numpy as np
import cPickle
import os
import argparse
from model import SRNN

###########################################################################################
#                                 SET SOME PARAMETERS                                     #
###########################################################################################
parser = argparse.ArgumentParser(description="Human activities && object affordance \
                                             detection and anticipation with srnn")

parser.add_argument('--data', type=str, default='./dataset',
                    help='location of the dataset')

parser.add_argument('--ptc', type=str, default='./checkpoints_anticipation',
                    help='location of the checkpoints files')

parser.add_argument('--cpf', type=str, default='detection_500_epochs.net',
                    help='the name of a specified checkpoint file')        

args = parser.parse_args()           

###########################################################################################
#                                   LOAD THE DATAS                                        #
###########################################################################################

test_data = cPickle.load(open('{}/test_data.pik'.format(args.data)))	
Y_h = test_data['labels_human']      #25..(12,1)
Y_h_a = test_data['labels_human_anticipation']      #25..(12,1)
X_h_d = test_data['features_human_disjoint']      #25..(12, 1, 790)
X_h_s = test_data['features_human_shared']          #25..(12, 1, 400)

Y_o = test_data['labels_objects']
Y_o_a = test_data['labels_objects_anticipation']
X_o_d = test_data['features_objects_disjoint']
X_o_s = test_data['features_objects_shared']

###########################################################################################
#                                GET LABEL AND EQUALS LIST                                #
###########################################################################################

def get_label(output, label):
    prob = torch.exp(output)
    top_p, top_class = prob.topk(1)
    equals = top_class == label.view(top_class.shape)
    return top_class, equals

###########################################################################################
#                                  CALCUALTE THE ACCURACY                                 #
###########################################################################################

def get_acc(equal_list):
    s, n = [0, 0]
    for equal in equal_list:
        s += np.array(equal).sum()
        n += len(equal)
    return s/n

###########################################################################################
#                                         RUNNING                                         #
###########################################################################################

# upload the checkpoints
with open('{}/{}'.format(args.ptc, args.cpf)) as f:
    checkpoints = torch.load(f)
# generate the network with checkpoints
net = SRNN(checkpoints['h_n_labels'], checkpoints['o_n_labels'], checkpoints['lstm1_input_size'],
           checkpoints['lstm2_input_segment_size'], checkpoints['n_hidden1'], checkpoints['n_hidden2'])
net.load_state_dict(checkpoints['state_dict'])
# set running device model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("running on {}".format(device))

net.to(device)
# initialize hidden states,and to device
h_hiddens = net.init_hidden(1, device)
o_hiddens = net.init_hidden(1, device)
hs = [h_hiddens, o_hiddens]

pre_h, pre_o, acc_h, acc_o = [], [], [], []
# calculate the result
for x_h_s, x_h_d, y_h, x_o_s, x_o_d, y_o in zip(X_h_s, X_h_d, Y_h, X_o_s, X_o_d, Y_o):
    # set running model
    net.eval()
    # put element of data into running device
    dataset = [x_h_s, x_h_d, y_h, x_o_s, x_o_d, y_o]
    for i, data in enumerate(dataset):
        dataset[i] = torch.from_numpy(data).to(device)
    x_h_s, x_h_d, y_h, x_o_s, x_o_d, y_o = dataset
    # feed data into trained model
    output, hs = net([x_h_s, x_h_d], [x_o_s, x_o_d], hs[0], hs[1])
    # get the label which has the max probability
    pre_h_label, equal_h = get_label(output[0], y_h)
    pre_o_label, equal_o = get_label(output[1], y_o)
    # preserve the prediction
    pre_h.append(pre_h_label)
    pre_o.append(pre_o_label)
    acc_h.append(equal_h)
    acc_o.append(equal_o)
# calculate the accuracy
accuracy_h = get_acc(acc_h)
accuracy_o = get_acc(acc_o)
print(accuracy_h)
print(accuracy_o) 




