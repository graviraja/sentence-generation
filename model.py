'''This code contains the model definition.

'''
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    '''This implements the encoder part.

    '''
    def __init__(self, input_dim, embedding_dim, hidden_dim, latent_dim, num_layers, bidrectional, dropout, device):
        '''
        Args:
            input_dim: A integer indicating the size of input dim (vocabulary size).
            embedding_dim: A integer indicating the size of embedding dim.
            hidden_dim: A integer indicating the hidden dimension.
            latent_dim: A integer indicating the latent vectors dimension.
            num_layers: A integer indicating the number of layers in rnn.
            bidrectional: A boolean indicating whether the rnn is bidirectional or not.
            dropout: A float indicating the amount of dropout between rnn layers.
        '''
        super().__init__()

        self.bidirectional = bidrectional
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, bidrectional=bidrectional, dropout=dropout, batch_first=True)

        self.hidden_factor = (2 if bidrectional else 1) * num_layers
        self.mu = nn.Linear(self.hidden_factor * hidden_dim, latent_dim)
        self.var = nn.Linear(self.hidden_factor * hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, input_lengths):
        # input is of shape [batch_size, max_len]
        # input_lengths contains [length_of_each_sequence_in_batch]
        batch_size = input.shape[0]

        # sort the input data
        sorted_lengths, sorted_index = torch.sort(input_lengths, descending=True)
        input = input[sorted_index]

        embed = self.embedding(input)
        embed = self.dropout(embed)
        # embed is of shape [batch_size, batch_size, embedding_size]

        # applying packing method for embeddings
        # expects input_lengths as a list, so convert lengths tensor to list
        packed_input = rnn_utils.pack_padded_sequence(embed, sorted_lengths.data.tolist(), batch_first=True)

        # apply rnn on packed input
        outputs, hidden = self.rnn(packed_input)
        # outputs is of shape [max_len, batch_size, hidden_dim * num_dir] => [max_len, batch_size, 2] if bidirectional rnn.
        #                                                                 => [max_len, batch_size, 1] if unidirectional rnn.
        # hidden is of shape [num_layers * num_dir, batch_size, hidden_dim] => [num_layers * 2, batch_size, hidden_dim] if birectional rnn.
        #                                                                   => [num_layers, batch_size, hidden_dim] if unidirectional rnn.

        # we consider only hidden states for creating latent space vectors.
        # for latent parameters creation, input must in form of [batch_size, self.hidden_factor * hidden_dim]
        hidden = hidden.view(batch_size, -1)

        # reparameterization
        mean = self.mu(hidden)
        var = self.var(hidden)

        # mean is of shape [batch_size, latent_dim]
        # var is of shape [batch_size, latent_dim]
        return mean, var


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_dim, latent_dim, hidden_dim, num_layers, bidirectional, dropout, word_dropout_rate, token_ids, device):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.word_dropout_rate = word_dropout_rate
        self.device = device
        self.sos_idx = token_ids['<sos>']
        self.pad_idx = token_ids['<pad>']
        self.eos_idx = token_ids['<eos>']
        self.unk_idx = token_ids['<unk>']

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.hidden_factor = (2 if bidrectional else 1) * num_layers
        self.latent_to_hidden = nn.Linear(latent_dim, self.hidden_factor * hidden_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

        self.output_to_vocab = nn.Linear(hidden_dim * (2 if bidirectional else 1), input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, input_lengths, latent_vector):
        # input is of shape [batch_size, max_len]
        # input_lengths contains [length_of_each_sequence_in_batch]
        # latent_vector is of shape [batch_size, latent_dim]
        batch_size = latent_vector.shape[0]

        # sort the input data
        sorted_lengths, sorted_index = torch.sort(input_lengths, descending=True)
        input = input[sorted_index]

        decoder_hidden = self.latent_to_hidden(latent_vector)
        # decoder_hidden shape is [batch_size, self.hidden_factor * hidden_dim]

        # reshape the decoder_hidden into [self.hidden_factor, batch_size, hidden_dim]
        #                              => [num_layers * num_dir, batch_size, hidden_dim]
        if self.bidirectional or self.num_layers > 1:
            decoder_hidden = decoder_hidden.view(self.hidden_factor, batch_size, self.hidden_dim)
        else:
            # decoder_hidden shape in case of single layer, unidirectional => [1, batch_size, hidden_dim]
            decoder_hidden = decoder_hidden.unsqueeze(0)

        # input to the decoder
        decoder_input = input.clone()
        # this is we where we implement the word-dropout method mentioned in the paper
        if self.word_dropout_rate > 0:
            # randomly replace the decoder input with <unk> token
            probs = torch.rand(input.size()).to(self.device)
            # we replace only the actual words data not the sos, eos, pad token data
            probs[(input.data - self.sos_idx) * (input.data - self.pad_idx) * (input.data - self.eos_idx) == 0] = 1
            decoder_input[probs < self.word_dropout_rate] = self.unk_idx
        embed = self.embedding(decoder_input)
        embed = self.dropout(embed)

        packed_input = rnn_utils.pack_padded_sequence(embed, sorted_lengths.data.tolist(), batch_first=True)

        outputs, hidden = self.rnn(packed_input)

        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        packed_input = padded_outputs.contiguous()
        b, s, _ = padded_outputs.size()

        vocab_outputs = nn.functional.log_softmax(self.output_to_vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        vocab_outputs = vocab_outputs.view(b, s, self.embedding.num_embeddings)
        vocab_outputs = vocab_outputs.permute(1, 0, 2)
        return vocab_outputs

    def inference(self):
        pass


class SentenceGenerator(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, input_lengths):
        mean, var = self.encoder(input, input_lengths)

        std = torch.exp(var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(mean)

        outputs = self.decoder(input, input_lengths, x_sample)
        return outputs, mean, var
