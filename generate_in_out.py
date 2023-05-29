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

    load corpus

"""

corpus = json.load( open( output_path + 'corpus.json', 'r' ) )

print(f'corpus size: {len(corpus)}')
print('\n')

""" 

    split corpus in sequences

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

    save data

"""

with open( output_path + 'features.json', 'w' ) as f:
     
     json.dump( features, f )

with open( output_path + 'targets.json', 'w' ) as f:
    
    json.dump( targets, f )