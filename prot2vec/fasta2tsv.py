import numpy as np
import pandas as pd
from sys import argv
from pandas import DataFrame

# Turns a fasta file into a tsv file for easier manipulation

def betweenOsOx(text):

    """Fasta files are downloaded from Uniprot. Organism names are shown between the strings \"OS=\" and \"OX=\" """
    for i in range(len(text)):
        if text[i:i+3] == 'OS=':
            break
    for k in range(i,len(text)):
        if text[k:k+3] == 'OX=':
            break
    return(text[i+3:k-1].replace(' ','_'))

uniprot = open(argv[1],'r').readlines()
headerIndices = []

# Sequences in fasta format have new lines every 40 amino acids, and thus must be removed when constructing a "sentence (i.e. protein)" of "words (i.e. amino acids)"

for i in range(len(uniprot)):
    uniprot[i] = uniprot[i].replace('\n','')
    if '>' in uniprot[i]: 
        headerIndices.append(i)
headerIndices.append(len(uniprot)) 
print(len(headerIndices))

protein_array = np.zeros((len(headerIndices), 2), dtype=object)
for i in range(len(headerIndices)-1):
    protein_array[i,0] = betweenOsOx(uniprot[headerIndices[i]])
    protein_array[i,1] = ''.join(uniprot[headerIndices[i]+1:headerIndices[i+1]])

proteinDF = DataFrame(protein_array)
proteinDF.columns = ['organism','protein_sequence']
proteinDF.to_csv(argv[1].replace('.fasta','.tsv'), index = False, sep='\t')
