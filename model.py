'''This code contains the model definition.

'''
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    '''This implements the encoder part.

    '''
    def __init__(self, input_dim, embedding_dim, hidden_dim, latent_dim, n_layers, bidrectional, dropout):
        '''
        Args:
            input_dim: A integer indicating the size of input dim (vocabulary size).
            embedding_dim: A integer indicating the size of embedding dim.
            hidden_dim: A integer indicating the hidden dimension.
            latent_dim: A integer indicating the latent vectors dimension.
            n_layers: A integer indicating the number of layers in rnn.
            bidrectional: A boolean indicating whether the rnn is bidirectional or not.
            dropout: A float indicating the amount of dropout between rnn layers.
        '''
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidrectional=bidrectional, dropout=dropout)
        
        self.hidden_factor = (2 if bidrectional else 1) * n_layers
        self.mu = nn.Linear(self.hidden_factor * hidden_dim, latent_dim)
        self.var = nn.Linear(self.hidden_factor * hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input is of shape [max_len, batch_size]

        embed = self.embedding(input)
        # embed is of shape [max_len, batch_size, embedding_size]
        embed = self.dropout(embed)

        outputs, hidden = self.rnn(embed)
        # outputs is of shape [max_len, batch_size, hidden_dim * num_dir] => [max_len, batch_size, 2] if bidirectional rnn.
        #                                                                 => [max_len, batch_size, 1] if unidirectional rnn.
        # hidden is of shape [num_layers * num_dir, batch_size, hidden_dim] => [num_layers * 2, batch_size, hidden_dim] if birectional rnn.
        #                                                                   => [num_layers, batch_size, hidden_dim] if unidirectional rnn.

        # we consider only final hidden states for creating latent space vectors.


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class SentenceGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
