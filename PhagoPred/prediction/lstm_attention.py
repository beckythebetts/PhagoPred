import torch
   
class LSTM_Attention(torch.nn.Module):

    def __init__(self, input_size, lstm_hidden_size, lstm_dropout, attn_hidden_size, lstm_proj_size=0):
        """
        Bi-directional LSTM. Hidden state at each time step input to simple additive attention mechanism consisting of NN with one hidden layer outputting single attention weight for each time step. 
        All hidden states summed, weighted by attention weight, and binary classification performed by NN.
        Args:
            input_size (int): length of input (number of features per time step)
            lstm_hidden_size (int):
            lstm_dropout:
            attn_hidden_size (int): 
            lstm_proj_size (int): if > 0, projects output of lstm to proj_size with additional linear transformation
        """
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size, lstm_hidden_size, dropout=lstm_dropout, bidirectional= True, proj_size=lstm_proj_size, batch_first=False)
        self.attention1 = torch.nn.Linear(in_features=lstm_hidden_size, out_features=attn_hidden_size)
        self.attention2 = torch.nn.Linear(in_features=attn_hidden_size, out_features=1)
        self.classifier = torch.nn.Linear(in_features=lstm_hidden_size, out_features=2)

    def forward(self, x):
        """
        Args:
            x [Sequence length, batch_size, num_features]
        """
        lstm_out, _ = self.lstm(x) # lstm_out [Sequence length, batch_size, 2*lstm_hidden_size]
        attn_hidden = torch.nn.functional.relu(self.attention1(lstm_out)) # attn_hidden [sequence length, batch_size, attn_hidden_size]
        attn_weights = torch.nn.functional.softmax(self.attention2(attn_hidden), dim=0) # attn_weights[sequence length, batch_size, 1]
        context_vector = torch.sum(attn_weights*lstm_out, dim=0) # context_vector [batch_size, 2*lstm_hidden_size]
        output = self.classifier(context_vector) #output [batch_size]
        return output, attn_weights




