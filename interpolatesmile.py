import h5py
import numpy
import os
import pandas
from functools import reduce
from rdkit import Chem
import argparse
from molecules.model import MoleculeVAE
from molecules.utils import decode_smiles_from_indexes
from molecules.utils import one_hot_array, one_hot_index

source = 'C=CCc1ccc(OCC(=O)N(CC)CC)c(OC)c1'
dest = 'C=C(C)CNc1ccc([C@H](C)C(=O)O)cc1'
latent_dim = 292
steps = 100
width = 120
length = 500000

# def get_arguments():
#     parser = argparse.ArgumentParser(description='Interpolate from source to dest in steps')
#     parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
#     parser.add_argument('model', type=str, help='Trained Keras model to use.')
#     parser.add_argument('--source', type=str, default=SOURCE,
#                         help='Source SMILES string for interpolation')
#     parser.add_argument('--dest', type=str, default=DEST,
#                         help='Source SMILES string for interpolation')
#     parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
#                         help='Dimensionality of the latent representation.')
#     parser.add_argument('--width', type=int, default=WIDTH,
#                         help='Dimensionality of the latent representation.')
#     parser.add_argument('--steps', type=int, default=STEPS,
#                         help='Number of steps to take while interpolating between source and dest')
#     return parser.parse_args()

def interpolate(source, dest, steps, charset, model, latent_dim, width):
    source_just = source.ljust(width)
    dest_just = dest.ljust(width)
    print(source_just)
    print(dest_just)
    one_hot_encoded_fn = lambda row: map(lambda x: one_hot_array(x, len(charset)),
                                                one_hot_index(row, charset))
    source_encoded = numpy.array(map(one_hot_encoded_fn, source_just))
    source_x_latent = model.encoder.predict(source_encoded.reshape(1, width, len(charset)))
    dest_encoded = numpy.array(map(one_hot_encoded_fn, dest_just))
    dest_x_latent = model.encoder.predict(dest_encoded.reshape(1, width, len(charset)))

    step = (dest_x_latent - source_x_latent)/float(steps)
    results = []
    for i in range(steps):
        item = source_x_latent + (step  * i)
        sampled = model.decoder.predict(item.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset)
        results.append( (i, item, sampled) )

    return results

def main():
    # args = get_arguments()

    # if os.path.isfile(args.data):
    #     h5f = h5py.File(args.data, 'r')
    #     charset = list(h5f['charset'][:])
    #     h5f.close()
    # else:
    #     raise ValueError("Data file %s doesn't exist" % args.data)
    data = pandas.read_hdf('data/smiles_500k.h5', 'table')
    keys = data['structure'].map(len) < 121
    if length <= len(keys):
        data = data[keys].sample(n = length)
    else:
        data = data[keys]
    smiles = data['structure'].map(lambda x: list(x.ljust(120)))
    charset = list(reduce(lambda x, y: set(y) | x, smiles, set()))
    print(charset)
    print(len(charset))
    model = MoleculeVAE()
    if os.path.isfile('data/model_500k.h5'):
        model.load(charset, 'data/model_500k.h5', latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % 'data/model_500k.h5')

    results = interpolate(source, dest, steps, charset, model, latent_dim, width)
    for result in results:
        m = Chem.MolFromSmiles(result[2])
        if m != None:
            print(result[0], result[2])
        else:
            continue


if __name__ == '__main__':
    main()
