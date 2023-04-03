
from kgutils import generate_imprecise_kg, generate_imcomplete_kg

def eval_by_incorrect(kg, lpmodel):

    # step 1. corrupt a KG by taking a subset of triples from kg
    # step 2. generating some incorrect triples 
    # step 3. inject the incorrect triples into the corrupted KG
    # step 4. train the lpmodel on the injected kg
    # step 5. try to recover the taken-out triples by the trained lpmodel
    pass

def eval_by_missing(kg, lpmodel):
    # step 1. corrupt a KG by taking a subset of triples from kg
    # step 2. train the lpmodel on the injected kg
    # step 3. try to recover the taken-out triples by the trained lpmodel
    pass

def evaluate(kg, lpmodel, strategy='incorrect'):
    '''
    Take a KG as input, return the metrics about indicating the
    quality of this KG.
    '''
    if strategy == 'incorrect':
        return eval_by_incorrect(kg, lpmodel)
    elif strategy == 'missing':
        return eval_by_missing(kg, lpmodel)
    else:
        raise ValueError(f'"{strategy}" is not supported!')
    pass
