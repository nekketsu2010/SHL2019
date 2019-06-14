import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.version import cuda


class SimpleLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  # outは3Dtensorになるのでdim=2

    def forward(self, input, h, c):
        # nn.RNNは系列をまとめて処理できる
        # outputは系列の各要素を入れたときの出力
        # hiddenは最後の隠れ状態（=最後の出力） output[-1] == hidden[0]
        output, (h, c) = self.lstm(input, (h, c))

        # RNNの出力がoutput_sizeになるようにLinearに通す
        output = self.out(output)

        # 活性化関数
        output = self.softmax(output)

        return output, (h, c)

    def initHidden(self):
        # 最初に入力する隠れ状態を初期化
        # LSTMの場合は (h, c) と2つある
        # (num_layers, batch, hidden_size)
        h = Variable(torch.zeros(1, 1, self.hidden_size))
        c = Variable(torch.zeros(1, 1, self.hidden_size))
        if cuda:
            h = h.cuda()
            c = c.cuda()
        return (h, c)