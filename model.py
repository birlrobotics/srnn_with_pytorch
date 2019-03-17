import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class SRNN(nn.Module):
    def __init__(self, h_n_labels, o_n_labels, lstm1_input_size, lstm2_input_segment_size, n_hidden1=128, 
                                              n_hidden2=256, n_layers=2, drop_prob = 0.2 ):
        super(SRNN, self).__init__()
        
        self.h_n_labels = h_n_labels
        self.o_n_lables = o_n_labels
        self.lstm1_input_size = lstm1_input_size
        self.lstm2_input_segment_size = lstm2_input_segment_size
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.input_size1 = lstm1_input_size
        self.input_size2 = lstm2_input_segment_size + n_hidden1
        # define the LSTMs:for human_activities and object_affordance
        self.h_lstm = [nn.LSTM(self.input_size1 if n ==0 else self.input_size2[0], n_hidden1 if n != n_layers-1 else n_hidden2,
                                                                        1, batch_first=False ) for n in range(n_layers)]

        self.o_lstm = [nn.LSTM(self.input_size1 if n ==0 else self.input_size2[1], n_hidden1 if n != n_layers-1 else n_hidden2,
                                                                        1, batch_first=False ) for n in range(n_layers)]
        self.h_lstm = nn.ModuleList(self.h_lstm)
        self.o_lstm = nn.ModuleList(self.o_lstm)
        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        # define the final fully-connected output layer
        self.fc1 = nn.Linear(n_hidden2, h_n_labels)
        self.fc2 = nn.Linear(n_hidden2, o_n_labels)

    def forward(self, h_inputs, o_inputs, h_hiddens, o_hiddens):
        '''Forward pass through the network.

           There are two branchs in this neural network:
                one is for the human_activities;
                the other one is for the object;

           h_inputs[0]/o_inputs[0] is the inputs of lstm1, 
           while the inputs of lstm2 is concat(h_inputs[1]/o_inputs[1], output of lstm1)
        '''
        result = []
        state = []
        raw_output = None
        n = 0
        for inputs, hiddens in zip((h_inputs,o_inputs),(h_hiddens,o_hiddens)):
            new_hidden = []
            # select lstm structure for human_activities or object affordance
            if n == 0:
                lstm = self.h_lstm
            else:
                lstm = self.o_lstm
            for i, lstm in enumerate(lstm):
                if i == 0:
                    input = inputs[i]
                else:
                    input = torch.cat([inputs[i], raw_output], dim=2)
                # get the outputs and new hidden state from the lstm
                raw_output, hidden = lstm(input, hiddens[i])
                # pass through a dropout layer
                raw_output = self.dropout(raw_output)
                # store the new hidden
                new_hidden.append(hidden)
            # Stack up LSTM outputs using view
            # you may need to use contiguous to reshape the output
            raw_output = raw_output.contiguous().view(-1, self.n_hidden2)
            # pass through fully-connected layer
            if n == 0:
                output = self.fc1(raw_output)
                n += 1
            else:
                output = self.fc2(raw_output)
            # pass through softmax layer
            output = F.log_softmax(output, dim=1)
            # get the output and hidden state 
            result.append(output)
            state.append(new_hidden)

        return result, state
        
    def init_hidden(self, batch_size, device):
        '''initializes hidden state'''
        weight = next(self.parameters()).data
        hidden = ((weight.new(1, batch_size, self.n_hidden1).zero_().to(device),
                  weight.new(1, batch_size, self.n_hidden1).zero_().to(device)),
                  (weight.new(1, batch_size, self.n_hidden2).zero_().to(device),
                  weight.new(1, batch_size, self.n_hidden2).zero_().to(device)))
        return hidden







        

