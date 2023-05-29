import numpy as np
from collections import Counter

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

    load notes

"""

notes_list = [ x for x in sorted(os.listdir( corpus_path )) if '.json' in x ]

print(f'notes list size: {len(notes_list)}')
print('\n')

""" 

    generate corpus

"""

corpus = []

for n in tqdm(notes_list):

    #print(f'loading {n}...')
    #print('\n')

    aux = json.load( open( corpus_path + n, 'r' ) )

    aux = list(aux.values())
    aux = [ x for sub_x in aux for x in sub_x ]

    #print(f'corpus size: {len(aux)}')
    #print(type(aux))

    corpus.extend( aux )

print(f'corpus size: {len(corpus)}')
print('\n')

with open( output_path + 'corpus.json', 'w' ) as f:
    json.dump( corpus, f )