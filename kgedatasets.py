
'''
Utilities for loading datasets
'''
import os
from collections import namedtuple
from typing import Dict, Iterable, Tuple

import numpy as np
import tensorflow as tf
from ampligraph.datasets import load_fb13, load_wn11
from ampligraph.evaluation import (generate_corruptions_for_fit,    
                                    generate_corruptions_for_eval)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.environ.get('AMPLIGRAPH_DATA_HOME'):
    os.environ['AMPLIGRAPH_DATA_HOME'] = os.path.join(
        ROOT_DIR, 'ampligraph_datasets'
    )


_Triple = Tuple[str, str, str]

def is_contain_unseen(train_trps: Iterable[_Triple], valid_or_test:Iterable[_Triple]) -> bool:
    '''
    Check if there are triples in valid or test set containing unseen 
    entities or relations that are not in the train set.
    
    Requires
    --------
        triples are in <head, rel, tail> format
    '''
    train_trps = np.array(train_trps)
    valid_or_test = np.array(valid_or_test)
    train_ents = set(train_trps[:, 0]).union(set(train_trps[:, 2]))
    train_rels = set(train_trps[:, 1])
    for trp in valid_or_test:
        if (trp[0] not in train_ents) or (trp[2] not in train_ents) or (trp[1] not in train_rels):
            return True
    return False


def filter_unseen(train_trps: Iterable[_Triple], valid_or_test:Iterable[_Triple]) -> np.ndarray:
    '''
    Filter out triples of valid or test set that contain unseen 
    entities or relations that are not in the train set.
    
    Requires
    --------
        triples are in <head, rel, tail> format
    '''
    train_trps = np.array(train_trps)
    valid_or_test = np.array(valid_or_test)
    train_ents = set(train_trps[:, 0]).union(set(train_trps[:, 2]))
    train_rels = set(train_trps[:, 1])
    new_trps = []
    for trp in valid_or_test:
        if (trp[0] not in train_ents) or (trp[2] not in train_ents) or (trp[1] not in train_rels):
            continue
        new_trps.append(trp)
    return np.array(new_trps)


class DatasetWrapper:
    '''Adapter to wrap a dataset in order to fit our experiments
    '''
    def __init__(self, name, X_train, X_valid, y_valid, X_test, y_test):
        self.name = name
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test

    @property
    def stats(self) -> str:
        '''Return number of train/valid/test/entities/relations'''
        ents = set(self.X_train[:, 0]).union(self.X_train[:, 2])
        rels = set(self.X_train[:, 1])
        return len(self.X_train), len(self.X_valid), len(self.X_test), len(ents), len(rels)


def get_fb13() -> DatasetWrapper:
    tmp = load_fb13()
    return DatasetWrapper('FB13k', tmp['train'], 
                          tmp['valid'], 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'], 
                          tmp['test_labels'].astype(np.int32))


def get_wn11() -> DatasetWrapper:
    tmp = load_wn11()
    return DatasetWrapper('WN11', tmp['train'], 
                          tmp['valid'], 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'], 
                          tmp['test_labels'].astype(np.int32))


def _load_yago39():
    yago_path = os.environ['AMPLIGRAPH_DATA_HOME'] + os.sep + 'yago39' + os.sep
    data = {}
    with open(yago_path + 'train_triple2id.txt', 'r') as f:
        lines = f.readlines()
        data['train'] = np.array([line.strip().split() for line in lines[1:]])
    data['train'][:, [1, 2]] = data['train'][:, [2, 1]]  # adjust the columns of rel and tail for yago dataset
    train_entities = set(data['train'][:, 0]).union(set(data['train'][:, 2]))
    # print(len(train_entities))

    with open(yago_path + 'valid_triple2id_positive.txt', 'r') as f:
        lines = f.readlines()
        tmp = []
        for line in lines[1:]:
            triple = line.strip().split()
            if (triple[0] in train_entities) and (triple[2] in train_entities):
                tmp.append(triple)
    data['valid'] = np.array(tmp)
    data['valid_labels'] = np.ones(len(tmp))

    with open(yago_path + 'valid_triple2id_negative.txt', 'r') as f:
        lines = f.readlines()
        tmp = []
        for line in lines[1:]:
            triple = line.strip().split()
            if (triple[0] in train_entities) and (triple[2] in train_entities):
                tmp.append(triple)
    data['valid'] = np.concatenate([data['valid'], np.array(tmp)])
    data['valid_labels'] = np.concatenate([data['valid_labels'], np.zeros(len(tmp))])
    
    data['valid'][:, [1, 2]] = data['valid'][:, [2, 1]]

    with open(yago_path + 'test_triple2id_positive.txt', 'r') as f:
        lines = f.readlines()
        tmp = []
        for line in lines[1:]:
            triple = line.strip().split()
            if (triple[0] in train_entities) and (triple[2] in train_entities):
                tmp.append(triple)
    data['test'] = np.array(tmp)
    data['test_labels'] = np.ones(len(tmp))

    with open(yago_path + 'test_triple2id_negative.txt', 'r') as f:
        lines = f.readlines()
        tmp = []
        for line in lines[1:]:
            triple = line.strip().split()
            if (triple[0] in train_entities) and (triple[2] in train_entities):
                tmp.append(triple)
    data['test'] = np.concatenate([data['test'], np.array(tmp)])
    data['test_labels'] = np.concatenate([data['test_labels'], np.zeros(len(tmp))])

    data['test'][:, [1, 2]] = data['test'][:, [2, 1]]

    return data


def get_yago39() -> DatasetWrapper:
    tmp = _load_yago39()
    return DatasetWrapper('YAGO39', tmp['train'].astype(np.int32), 
                          tmp['valid'].astype(np.int32), 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'].astype(np.int32), 
                          tmp['test_labels'].astype(np.int32))


def load_dataset(dirpath: str) -> Dict[str, np.ndarray]:
    '''Load a dataset from a directory

    Requires
    --------
    The directory should have these files:
    - train.txt
    - valid.txt
    - test.txt
    - entity2id.txt
    - relation2id.txt

    Returns
    -------
    a dict with 5 keys: 
        dict_keys(['train', 'valid', 'test', 'valid_labels', 'test_labels'])
    '''
    e2id = {}
    with open(os.path.join(dirpath, 'entity2id.txt'), 'r', encoding='utf8') as f:
        for line in f:
            ent, eid = line.split()
            e2id[ent] = int(eid)
    r2id = {}
    with open(os.path.join(dirpath, 'relation2id.txt'), 'r', encoding='utf8') as f:
        for line in f:
            rel, rid = line.split()
            r2id[rel] = int(rid)

    def _load_dataset(filepath: str) -> np.ndarray:
        '''load data from xxx.txt'''
        trps = []
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f:
                s, p, o = line.split()
                # print(s, p, o)
                if (s in e2id) and (p in r2id) and (o in e2id):
                    trp = [e2id[s], r2id[p], e2id[o]]
                    trps.append(trp)
        return np.array(trps).astype(np.int32)

    train = _load_dataset(os.path.join(dirpath, 'train.txt'))
    tmp_valid = _load_dataset(os.path.join(dirpath, 'valid.txt'))
    tmp_test = _load_dataset(os.path.join(dirpath, 'test.txt'))
    
    # filter out unseen ents and rels in valid and test (w.r.t. train)
    train_ents = set(train[:, 0]).union(set(train[:, 2]))
    train_rels = set(train[:, 1])
    valid = []
    for trp in tmp_valid:
        if trp[0] in train_ents and trp[1] in train_rels and trp[2] in train_ents:
            valid.append(trp)
    valid = np.array(valid)
    test = []
    for trp in tmp_test:
        if trp[0] in train_ents and trp[1] in train_rels and trp[2] in train_ents:
            test.append(trp)
    test = np.array(test)

    # positive labels
    valid_labels = np.ones(len(valid))
    test_labels = np.ones(len(test))

    # generate negative triples and the corresponding labels
    valid_neg = generate_corruptions_for_fit(valid).eval(session=tf.Session())
    valid = np.vstack([valid, valid_neg])
    test_neg = generate_corruptions_for_fit(test).eval(session=tf.Session())
    test = np.vstack([test, test_neg])
    valid_labels = np.concatenate([valid_labels, np.zeros(len(valid_labels))])
    test_labels = np.concatenate([test_labels, np.zeros(len(test_labels))])

    return {
        'train': train,
        'valid': valid,
        'test': test,
        'valid_labels': valid_labels,
        'test_labels': test_labels
    }


def get_dp50():
    dp50_path = os.environ['AMPLIGRAPH_DATA_HOME'] + os.sep + 'DBpedia50' + os.sep
    tmp = load_dataset(dp50_path)
    return DatasetWrapper('DBpedia50', tmp['train'].astype(np.int32), 
                          tmp['valid'].astype(np.int32), 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'].astype(np.int32), 
                          tmp['test_labels'].astype(np.int32))

def get_umls():
    umls_path = os.environ['AMPLIGRAPH_DATA_HOME'] + os.sep + 'UMLS' + os.sep
    tmp = load_dataset(umls_path)
    return DatasetWrapper('UMLS', tmp['train'].astype(np.int32), 
                          tmp['valid'].astype(np.int32), 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'].astype(np.int32), 
                          tmp['test_labels'].astype(np.int32))

def get_kinship():
    kinship_path = os.environ['AMPLIGRAPH_DATA_HOME'] + os.sep + 'Kinship' + os.sep
    tmp = load_dataset(kinship_path)
    return DatasetWrapper('Kinship', tmp['train'].astype(np.int32), 
                          tmp['valid'].astype(np.int32), 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'].astype(np.int32), 
                          tmp['test_labels'].astype(np.int32))

def get_nations():
    nations_path = os.environ['AMPLIGRAPH_DATA_HOME'] + os.sep + 'Nations' + os.sep
    tmp = load_dataset(nations_path)
    return DatasetWrapper('Nations', tmp['train'].astype(np.int32), 
                          tmp['valid'].astype(np.int32), 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'].astype(np.int32), 
                          tmp['test_labels'].astype(np.int32))


def get_yago_et():
    yagoet_path = os.environ['AMPLIGRAPH_DATA_HOME'] + os.sep + 'yago-et' + os.sep
    tmp = load_dataset(yagoet_path)
    return DatasetWrapper('YAGO_ET', tmp['train'].astype(np.int32), 
                          tmp['valid'].astype(np.int32), 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'].astype(np.int32), 
                          tmp['test_labels'].astype(np.int32))

def get_dbpedia_et():
    dbpediaet_path = os.environ['AMPLIGRAPH_DATA_HOME'] + os.sep + 'dbpedia-et' + os.sep
    tmp = load_dataset(dbpediaet_path)
    return DatasetWrapper('DBpedia_ET', tmp['train'].astype(np.int32), 
                          tmp['valid'].astype(np.int32), 
                          tmp['valid_labels'].astype(np.int32), 
                          tmp['test'].astype(np.int32), 
                          tmp['test_labels'].astype(np.int32))
                          

ExperimentDatasets = namedtuple('Datasets', [
    'fb13', 
    'wn18', 
    'yago39',
    'dp50',
    'umls',
    'kinship',
    'nations',
    'yago_et',
    'dbpedia_et',
])

def get_datasets() -> ExperimentDatasets:
    lst = []
    lst.append(get_fb13())
    lst.append(get_wn11())
    lst.append(get_yago39())
    lst.append(get_dp50())
    lst.append(get_umls())
    lst.append(get_kinship())
    lst.append(get_nations())
    lst.append(get_yago_et())
    lst.append(get_dbpedia_et())
    return ExperimentDatasets(*lst)

if __name__ == '__main__':
    get_datasets()
    
