'''This code contains the training process.

'''
import torch
import torch.nn as nn
import torch.optim as optim
from model import Encoder, Decoder, SentenceGenerator
from data import load_vocab

vocabulary = load_vocab()
INPUT_DIM = len(vocabulary.vocab)
OUTPUT_DIM = len(vocabulary.vocab)
HIDDEN_DIM = 300
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_DROPOUT = 0.4
DEC_DROPOUT = 0.4
LATENT_DIM = 400
ENC_LAYERS = 2
DEC_LAYERS = 2
ENC_BIDIRECTIONAL = True
DEC_BIDIRECTIONAL = False
WORD_DROPOUT_RATE = 0.3
token_ids = {
    '<sos>': vocabulary.vocab.stoi['<sos>'],
    '<eos>': vocabulary.vocab.stoi['<eos>'],
    '<pad>': vocabulary.vocab.stoi['<pad>'],
    '<unk>': vocabulary.vocab.stoi['<unk>']
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, LATENT_DIM, ENC_LAYERS, ENC_BIDIRECTIONAL, ENC_DROPOUT, device).to(device)
decoder = Decoder(INPUT_DIM, DEC_EMB_DIM, LATENT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_BIDIRECTIONAL, DEC_DROPOUT, WORD_DROPOUT_RATE, token_ids, device).to(device)
model = SentenceGenerator(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=token_ids['<pad>'])


def train(model, iterator, criterion, optimizer, clip):
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        lengths = batch.lengths

        optimizer.zero_grad()
        output, mean, var = model(src, lengths)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        rec_loss = criterion(output, trg)
        kl_loss = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
        loss = rec_loss + kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def validate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            lengths = batch.lengths

            output, mean, var = model(src, lengths)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            rec_loss = criterion(output, trg)
            kl_loss = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
            loss = rec_loss + kl_loss

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)
