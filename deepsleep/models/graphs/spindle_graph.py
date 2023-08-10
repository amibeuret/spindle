import torch.nn as nn
import torch.nn.functional as functional


class SpindleGraph(nn.Module):
    def __init__(self, input_dim, nb_class, dropout_rate):
        super(SpindleGraph, self).__init__()

        self.num_chan = input_dim[0]
        self.height = input_dim[1]
        self.width = input_dim[2]

        self.nb_class = nb_class
        self.prob = dropout_rate

        # Convolution parameters
        out_chan, k1, k2, s1, s2 = 30, 3, self.height, 1, 1
        conv_h = ((self.height + 2 * 0 - 1 * (k1 - 1) - 1) // s1) + 1
        conv_w = ((self.width + 2 * 0 - 1 * (k2 - 1) - 1) // s2) + 1

        # Maxpool parameters
        mk1, mk2, ms1, ms2 = 2, 1, 2, 1
        maxp_h = ((conv_h + 2 * 0 - 1 * (mk1 - 1) - 1) // ms1) + 1
        maxp_w = ((conv_w + 2 * 0 - 1 * (mk2 - 1) - 1) // ms2) + 1

        # FC1 parameters
        out_l = 1000

        self.convnet = nn.Conv2d(self.num_chan, out_chan,
                                 kernel_size=(k1, k2),
                                 stride=(s1, s2))
        self.max_pool = nn.MaxPool2d(kernel_size=(mk1, mk2),
                                     stride=(ms1, ms2))
        self.fc1 = nn.Linear(30*maxp_h*maxp_w, out_l)
        self.fc2 = nn.Linear(out_l, nb_class)
        self.logSoftMax = nn.LogSoftmax(dim=1)
        self.drop_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = functional.relu(self.convnet(x))
        x = self.max_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.drop_layer(x)
        x = functional.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = self.fc2(x)
        return self.logSoftMax(x)
