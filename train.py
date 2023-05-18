import numpy as np

import pretty_midi
import librosa
import scipy.io.wavfile as wavfile

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from tqdm import tqdm

import os

# viz
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

"""

    where am i?
    
"""

PATH = os.getcwd() + '/'
data_path = PATH + 'data/'
midi_path = data_path + '/nesmdb_midi/train/'
output_path = data_path + 'output/'
model_path = PATH + 'model/'

print(f'PATH: {PATH}')
print(f'data path: {data_path}')
print(f'midi path: {midi_path}')
print(f'output path: {output_path}')
print(f'model path: {model_path}')

"""

    device

"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'device: {device}')

"""

    global parameters

"""

batch_size = 100

lr = 1e-3

latent_dim = 2

"""

    load midi files

"""

sample_space = sorted(os.listdir( midi_path ))
sample_space = np.random.choice( sample_space, 1000 )

print(f'sample space size: {len(sample_space)}')

"""

    load midi files

"""

midi = {}
midi_error = []

for s in tqdm( sample_space ):

    try:
        
        # piano roll representation
        aux = pretty_midi.PrettyMIDI( midi_path + s ).get_piano_roll( fs=100 )

        if aux.shape[1] > 0:
            midi[ s ] = aux
        else: 
            midi_error.append(s)
        
    except:
        
        #print(f'error: {s}')
        midi_error.append(s)

print(f'sample space: {len(midi)}')
print(f'corrupted files: {len(midi_error)}')

"""

    data loader

"""

class AudioDataset( Dataset ):
    
    def __init__(self, midi_files):
        
        self.midi_files = midi_files

    def __len__(self):
        
        return len( self.midi_files )

    def __getitem__( self, idx ):
        
        midi_file = self.midi_files[ idx ]
        
        # log-frequency spectogram
        log_spec = librosa.amplitude_to_db(
            librosa.feature.melspectrogram(
                y=None, 
                sr=100, 
                S=midi_file.T, 
                n_fft=1024, 
                hop_length=512, 
                power=2.0, 
                n_mels=128), 
            ref=1.0
        )
        # convert to pytorch tensor
        eps = 1e-38
        log_spec_db = torch.from_numpy( log_spec ).float()#.unsqueeze(0)  # add channel dimension
        log_spec_norm = ( log_spec_db - torch.min(log_spec_db) ) / ( torch.max(log_spec_db) - torch.min(log_spec_db) + eps)
        log_spec_norm = log_spec_norm.unsqueeze(0)

        return log_spec_norm
    
dataset = AudioDataset( list(midi.values()) )

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True
)

for i, spec_tensor in enumerate( dataloader ):
    
    print(f'batch {i}: {spec_tensor.size()}')
    print(f'batch variance: {spec_tensor.var()}')
    print(f'max, min: {spec_tensor.max()}, {spec_tensor.min()}')
    print('\n')

""" 

    vanilla vae

"""

class Encoder( nn.Module ):

    def __init__( self, input_dim, hidden_dim, latent_dim ):

        super( Encoder, self ).__init__()

        self.fc1 = nn.Linear( input_dim, hidden_dim )
        self.fc2 = nn.Linear( hidden_dim, hidden_dim )
        self.fc3 = nn.Linear( hidden_dim, latent_dim ) 
        self.fc4 = nn.Linear( hidden_dim, latent_dim )

        self.relu = nn.ReLU(0.2)
        self.training = True

    def forward( self, x ):

        x = self.relu( self.fc1( x ) )
        x = self.relu( self.fc2( x ) )

        mu = self.fc3( x )
        logvar = self.fc4( x )

        return mu, logvar
    
class Decoder( nn.Module ):

    def __init__( self, latent_dim, hidden_dim, output_dim ):

        super( Decoder, self ).__init__()

        self.fc1 = nn.Linear( latent_dim, hidden_dim )
        self.fc2 = nn.Linear( hidden_dim, hidden_dim )
        self.fc3 = nn.Linear( hidden_dim, output_dim )

        self.relu = nn.ReLU(0.2)

    def forward( self, x ):

        x = self.relu( self.fc1( x ) )
        x = self.relu( self.fc2( x ) )

        x = self.fc3( x )

        x_hat = torch.sigmoid( x )

        return x_hat
    
class VAE( nn.Module ):

    def __init__( self, Encoder, Decoder ):

        super( VAE, self ).__init__()

        self.encoder = Encoder
        self.decoder = Decoder

    def reparametrization( self, mu, logvar ):

        eps = torch.randn_like( logvar ).to(device)
        z = mu + (eps * torch.exp(0.5 * logvar))

        return z
    
    def forward( self, x ):

        mean, logvar = self.encoder( x )
        z = self.reparametrization( mean, logvar )

        x_hat = self.decoder( z )

        return x_hat, mean, logvar

""" 

    model

"""

encoder = Encoder( input_dim=128*128, hidden_dim=512, latent_dim=latent_dim ).to(device)
decoder = Decoder( latent_dim=latent_dim, hidden_dim=512, output_dim=128*128 ).to(device)

model = VAE( encoder, decoder ).to(device)

print(model)

""" 

    loss function

"""

def loss_function( x_hat, x, mu, logvar ):

    reproduction_loss = F.binary_cross_entropy( x_hat, x, reduction='sum' )
    KLD = -0.5 * torch.sum( 1 + logvar - mu.pow(2) - logvar.exp() ) # KL divergence

    return reproduction_loss + KLD

BCE_loss = nn.BCELoss() # binary cross entropy loss
optimizer = Adam( model.parameters(), lr=lr )

""" 

    train

"""

epochs = 100
best_loss = float('inf')

loss_dict = {}
for epoch in range( epochs ):

    overall_loss = 0

    for i, spec_tensor in enumerate( dataloader ):

        x = spec_tensor.view(-1, 128*128)
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, logvar = model( x )

        loss = loss_function( x_hat, x, mean, logvar )

        loss.backward()

        overall_loss += loss.item()

        optimizer.step()

    average_loss = overall_loss / (batch_size * i)
    print(f'epoch: { epoch }, loss: { average_loss }')

    loss_dict[ epoch ] = average_loss

    if average_loss < best_loss:
        best_loss = average_loss
        torch.save( model.state_dict(), model_path + 'vae.pth' )