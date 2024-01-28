import os
import time
import gensim
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kmer", type=int, help="# of Amino Acids in Kmer")
parser.add_argument("-w", "--window", type=int, help="Window Size for SkipGram")
parser.add_argument("-v", "--vsize", type=int, help="Vector Size")
parser.add_argument("-e", "--epochs", type=int, help="# of Epochs to Run")
parser.add_argument("-s", "--seed", type=int, help="Seed for Reproducibility")
args = parser.parse_args()

valid_aa = set(['M','H','Q','P','R','L','D','E','A','G','V','Y','C','W','F','N','K','T','S','I'])
print("Number of valid amino acids:", len(list(valid_aa)))

mammalian = pd.read_csv('uniprot_sprot_mammalian.tsv', sep='\t')
unique_organisms = mammalian['organism'].drop_duplicates().values
mammalian_proteins = mammalian['protein_sequence'].values

aa_count = 0
for i in range(len(mammalian_proteins)):
    aa_count += len(mammalian_proteins[i])

mammalian_proteins_valid = []

for i in range(len(mammalian_proteins)):
    isIn = True
    for j in range(len(mammalian_proteins[i])):
        if mammalian_proteins[i][j] not in valid_aa:
            isIn = False
            break
    if isIn == True:
        mammalian_proteins_valid.append(mammalian_proteins[i])
print("Number of peptides before deduplicating:", len(mammalian_proteins_valid))
mammalian_proteins_valid = list(set(mammalian_proteins_valid))
print("Number of peptides after deduplicating:", len(mammalian_proteins_valid))
aa_counts = 0
for term in mammalian_proteins_valid:
    aa_counts += len(term)
print("Total Number of Single Amino Acids in Corpus:", aa_counts)

def split_kmer(liszt, k):
    """Separates a list of n polypeptide sequences into a list of n*k k-grams, accounting for frameshift.
    For sequences of length not evenly divisible by k, remove the last letter(s) until the length is divisible by k"""
    new_list = []
    for x in range(len(liszt)):
        peptide = liszt[x]
        for y in range(k):
            subpeptide = peptide[y:len(peptide)]
            subpeptide_list = [subpeptide[z:z+k] for z in range(len(subpeptide)) if z % k ==0 and len(subpeptide[z:z+k])== k]
#             print(y,subpeptide, subpeptide_list)
            new_list.append(subpeptide_list)
    return(new_list)

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = get_tmpfile("{}_epoch{}".format(self.path_prefix, self.epoch))
        model.save('{}.model'.format(output_path))
        model.wv.save_word2vec_format('{}.w2v'.format(output_path))
        self.epoch += 1

parent = "/home/groups/song/songlab2/jimmyy2/prot2vec/prot2vec_{}gram_window{}_vsize{}_seed{}".format(args.kmer, args.window, args.vsize, args.seed)

if not os.path.exists(parent):
    os.makedirs(parent)
epoch_saver = EpochSaver('{}/prot2vec_{}gram'.format(parent, args.kmer))
mammalian_aa_valid = split_kmer(mammalian_proteins_valid, k=args.kmer)

t1 = time.time()
model = gensim.models.Word2Vec(mammalian_aa_valid, size=args.vsize, window=args.window, min_count=2, workers=10, sg=1, seed=args.seed)
print(time.time()-t1)

model.train(sentences = mammalian_aa_valid, total_examples=len(mammalian_aa_valid),epochs=args.epochs,  callbacks = [epoch_saver])
