import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

import json

from tqdm import tqdm

import os

"""

    where am i?
    
"""

PATH = os.getcwd() + '/'
data_path = PATH + 'data/'
train_path = data_path + '/nesmdb_midi/train/'
output_path = data_path + 'output/'
corpus_path = output_path + 'corpus/'

print(f'PATH: {PATH}')
print(f'data path: {data_path}')
print(f'train path: {train_path}')
print(f'output path: {output_path}')
print(f'corpus path: {corpus_path}')
print('\n')

"""

    device

"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'device: {device}')

"""

    load corpus

"""

corpus = json.load( open( output_path + 'corpus.json', 'r' ) )

print(f'corpus raw size: {len(corpus)}')
print(f'unique chars: {len(set(corpus))}')
print('\n')

corpus = corpus[:144574]
n_vocab = len(set(corpus))

print(f'corpus cut size: {len(corpus)}')
print(f'unique chars: { n_vocab }')
print('\n')

""" 

    split and encode corpus in sequences

"""

corpus_chars = sorted( list( set( corpus ) ) )
mapping = dict((c, i) for i, c in enumerate( corpus_chars ))

length = 100

features = []
targets = []

for i in tqdm(range( 0, len(corpus) - length, 1 )):

    input = corpus[ i:i + length ]
    output = corpus[ i + length ] 

    features.append( [ mapping[char] for char in input ] )
    targets.append( mapping[output] )

print(f'corpus sequences: {len(features)}')
print(f'targets: {len(targets)}')
print('\n')

"""  

    input format: [ sample, time steps, features ]

"""

X = torch.tensor( features, dtype=torch.float32 ).reshape( len(features), length, 1 )
X = X / float( n_vocab )

y = torch.tensor( targets, dtype=torch.float32 )

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')

""" 

    lstm

"""

class bitLSTM( nn.Module ):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM( input_size=1, hidden_size=512, num_layers=1, batch_first=True )
        self.dropout = nn.Dropout( p=0.2 )
        self.fc = nn.Linear( in_features=512, out_features=n_vocab )

    def forward(self, x):

        x, _ = self.lstm( x )
        x = x[ : , -1 , : ]
        x = self.dropout( x )
        x = self.fc( x )

        return x
    
model = bitLSTM().to(device)

print(f'model: {model}')

"""

    train

"""

num_epochs = 100
batch_size = 128

optimizer = torch.optim.Adam( model.parameters(), lr=0.001 )
criterion = nn.CrossEntropyLoss( reduction='sum' )
dataloader = DataLoader( TensorDataset( X, y ), batch_size=batch_size, shuffle=True )

best_model = None
best_loss = np.inf
dict_loss = {}

for epoch in range( num_epochs ):

    print('training...')

    model.train()

    for X_batch, y_batch in tqdm(dataloader):

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print('evaluating...')
    model.eval()
    loss = 0
    with torch.no_grad():

        for X_batch, y_batch in tqdm(dataloader):

            y_pred = model(X_batch)
            loss += criterion(y_pred, y_batch.long()).item()

        if loss < best_loss:

            best_loss = loss
            best_model = model.state_dict()

        dict_loss[epoch] = loss
        print(f'epoch: {epoch} | loss: {loss}')

with open( output_path + 'bitLSTM_loss.json', 'w' ) as f:
    json.dump( dict_loss, f )

torch.save( [best_model, mapping], output_path + 'bitLSTM.pt' )