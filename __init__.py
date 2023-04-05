from copy import deepcopy
import random
import tensorflow as tf
import numpy as np
from ampligraph.latent_features import TransE, ComplEx
from ampligraph.datasets import load_fb15k
from ampligraph.evaluation import (generate_corruptions_for_fit,
                                   create_mappings,
                                   to_idx, evaluate_performance,
                                   mrr_score, hits_at_n_score)
from ampligraph.utils import save_model



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

def inject_incorrect(id_train, proportion=0.3):
    sampled = random.sample(id_train.tolist(), int(len(id_train)*proportion))
    generated_incorrect = sample_negative(np.array(sampled, dtype=np.int32))
    # replace the original sampled triples with incorrect ones
    original = tuple(map(tuple, id_train))
    sampled = tuple(map(tuple, sampled))
    remaind = list(set(original) - set(sampled))
    return np.concatenate([remaind, generated_incorrect])



fb15k = load_fb15k()
id_train, id_valid, id_test = numerise_kg(fb15k['train'], fb15k['valid'], fb15k['test'])
injected_train = inject_incorrect(id_train)
print(len(injected_train))



hyperparams = {
        'verbose': True,
        # taken from https://arxiv.org/abs/1912.10000
        'k': 100,
        'optimizer': 'adam',
        'loss': 'nll',
        'eta': 20,
        'optimizer_params': {'lr': 1e-4},
        'epochs': 500,
        'early_stopping': True,
        'early_stopping_params': {
            'x_valid': id_valid
        }
    }

transe1 = TransE(**hyperparams)
transe2 = deepcopy(transe1)

transe1.fit(id_train)
filter_original = np.concatenate((id_train, id_valid, id_test))
ranks1 = evaluate_performance(id_test, model=transe1,
                            filter_triples=filter_original,
                            corrupt_side='s+o')
print(f'Original: mrr={mrr_score(ranks=ranks1)}, Hit@3={hits_at_n_score(ranks1, 3)}')


transe2.fit(injected_train)
filter_injected = np.concatenate((injected_train, id_valid, id_test))
ranks2 = evaluate_performance(id_test, model=transe2,
                            filter_triples=filter_original,
                            corrupt_side='s+o')
print(f'Injected: mrr={mrr_score(ranks=ranks2)}, Hit@3={hits_at_n_score(ranks2, 3)}')

save_model(transe1, './transe1.pt')
save_model(transe2, './transe2.pt')