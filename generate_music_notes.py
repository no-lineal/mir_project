from music21 import converter, instrument, note, chord

from tqdm import tqdm

import json

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

    load midi files

"""

train_space = sorted(os.listdir( train_path ))

print(f'sample space size: {len(train_space)}')
print('\n')

"""

    split batch

"""

batch_size = 100

train_batch = [ train_space[i:i+batch_size] for i in range(0, len(train_space), batch_size) ]

print(f'batch size: {len(train_batch)}')
print('\n')

n = 0
for b in train_batch:

    batch_dict = {}
    notes = []
    pick = None

    for m in tqdm(b):

        aux = converter.parse( train_path + m )

        songs = instrument.partitionByInstrument( aux )

        for part in songs.parts:

            pick = part.recurse()

            for element in pick:

                if isinstance( element, note.Note ):

                    notes.append( str( element.pitch ) )

                elif isinstance( element, chord.Chord ):

                    notes.append( '.'.join( str(n) for n in element.normalOrder ) )

        batch_dict[m] = notes

    with open( corpus_path + 'batch_' + str(n) + '.json', 'w' ) as f:

        json.dump( batch_dict, f )    

    n += 1