import argparse
import os
import random
from copy import deepcopy
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from ampligraph.latent_features import EmbeddingModel, TransE, ComplEx
from ampligraph.datasets import load_fb15k, load_fb15k_237, load_wn11, load_wn18, load_wn18rr, load_yago3_10
from ampligraph.utils import save_model
from ampligraph.evaluation import (create_mappings, to_idx, 
                                   generate_corruptions_for_fit,
                                   train_test_split_no_unseen,
                                   evaluate_performance,
                                   mrr_score, hits_at_n_score)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.environ.get('AMPLIGRAPH_DATA_HOME'):
    os.environ['AMPLIGRAPH_DATA_HOME'] = os.path.join(
        ROOT_DIR, 'ampligraph_datasets'
    )


parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_dir')
parser.add_argument('-o', '--output_dir')
args = parser.parse_args()
output_dir = args.output_dir
os.environ['AMPLIGRAPH_DATA_HOME'] = args.input_dir

print(os.environ['AMPLIGRAPH_DATA_HOME'])

def get_cls_name(obj):
    '''Get the name of the class of the object
    
    Example
    -------
    >>> l = [1, 2, 3]
    >>> get_cls_name(l)
    'list'
    '''
    return obj.__class__.__name__


def numerise_kg(train, valid, test):
    trps = np.concatenate([train, valid, test])
    rel_to_idx, ent_to_idx = create_mappings(trps)
    id_train = to_idx(train, ent_to_idx, rel_to_idx).astype(np.int32)
    id_valid = to_idx(valid, ent_to_idx, rel_to_idx).astype(np.int32)
    id_test = to_idx(test, ent_to_idx, rel_to_idx).astype(np.int32)
    return id_train, id_valid, id_test

def sample_negative(id_trps):
    assert id_trps.dtype == np.int32, f'dtype of input triples should be int32, not {id_trps.dtype}'
    neg = generate_corruptions_for_fit(id_trps).eval(session=tf.Session())
    return neg

def inject_incorrect(id_train, proportion=0.3) -> np.ndarray:
    sampled = random.sample(id_train.tolist(), int(len(id_train)*proportion))
    generated_incorrect = sample_negative(np.array(sampled, dtype=np.int32))
    # replace the original sampled triples with incorrect ones
    original = tuple(map(tuple, id_train))
    sampled = tuple(map(tuple, sampled))
    remaind = list(set(original) - set(sampled))
    return np.vstack([remaind, generated_incorrect])


def measure_kg(lpmodel, kg:np.ndarray, beta=0.1):
    '''
    Measure the quality of a KG via Link Prediction
    '''
    G, g = train_test_split_no_unseen(kg, test_size=int(len(kg)*beta))
    lpmodel.fit(G)
    ranks = evaluate_performance(g, model=lpmodel, filter_triples=kg)
    return {
        'mrr': mrr_score(ranks),
        'hit@1': hits_at_n_score(ranks, 1),
        'hit@3': hits_at_n_score(ranks, 5),
        'hit@10': hits_at_n_score(ranks, 10)
    }
    
def experiment(lpmodel, kg, alpha=0.3, beta=0.1, kgname='kg', expname=None):
    neg_kg = inject_incorrect(kg, alpha)
    model1 = deepcopy(lpmodel)
    model2 = deepcopy(lpmodel)

    if expname == None:
        expname = datetime.now().strftime('%m.%d-%H.%M.%S')
    exppath = os.path.join(output_dir, expname)
    
    res = pd.DataFrame({
        'original': measure_kg(model1, kg), 
        'injected': measure_kg(model2, neg_kg)
    })
    print(get_cls_name(lpmodel), kgname)
    print(res)
    if not os.path.exists(exppath):
        os.mkdir(exppath)
    with open(os.path.join(exppath,'report.txt'), 'a') as f:
        f.write(kgname+'\n============\n')
        f.write(res.to_latex())
    try:
        save_model(model1, os.path.join(exppath, f'{get_cls_name(model1)}-{kgname}_orig.pt'))
        save_model(model2, os.path.join(exppath, f'{get_cls_name(model2)}-{kgname}_neg.pt'))
    except TypeError as err:
        print(err)
    return 





hyperparams = {
        'verbose': True,
        # taken from https://arxiv.org/abs/1912.10000
        # 'k': 100,
        # 'optimizer': 'adam',
        # 'loss': 'nll',
        # 'eta': 20,
        'optimizer_params': {'lr': 1e-4},
        'epochs': 300,
    }

transe = TransE(**hyperparams)
comp = ComplEx(**hyperparams)

fb15k237 = load_fb15k_237()
wn18rr = load_wn18rr()
yago310 = load_yago3_10()
fb15k = load_fb15k()
wn18 = load_wn18()

fb15k['name'] = 'fb15k'
wn18['name'] = 'wn18'
fb15k237['name'] = 'fb15k237'
wn18rr['name'] = 'wn18rr'
yago310['name'] = 'yago310'


for ds in [fb15k, fb15k237, wn18, wn18rr, yago310]:
    id_train, id_valid, id_test = numerise_kg(ds['train'], ds['valid'], ds['test'])
    kg = np.vstack([id_train, id_test])

    experiment(comp, kg, kgname=ds['name'], expname='exp4-comp')
    experiment(transe, kg, kgname=ds['name'], expname='exp4-transe')

