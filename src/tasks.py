from pathlib import Path
from typing import List, Tuple
import pandas as pd
import random
import numpy as np

from missingadjunct.corpus import Corpus
from missingadjunct.utils import make_blank_sr_df
from src.utils import calc_sr_cores_from_spatial_model


from src.params import Params
from src.other_dsms.count import CountDSM
from src.other_dsms.w2vec import W2Vec
from src.other_dsms.rnn import RNN
from src.other_dsms.transformer import Transformer
from src.networks.ctn import CTN
from src.networks.lon import LON





def select_instrument(df_blank, dsm, instruments, params, step_bound, save_path):
# fill in blank data frame with semantic-relatedness scores
    df_results = df_blank.copy()
    for verb_phrase, row in df_blank.iterrows():
        verb, theme = verb_phrase.split()

        # score graphical models
        if isinstance(dsm, LON) or isinstance(dsm, CTN):
            scores = dsm.calc_sr_scores(verb, theme, instruments, step_bound)

        # score spatial models
        else:
            if params.composition_fn == 'native':  # use next-word prediction to compute sr scores
                scores = dsm.calc_native_sr_scores(verb, theme, instruments)
            else:
                scores = calc_sr_cores_from_spatial_model(dsm, verb, theme, instruments, params.composition_fn)

        # collect sr scores in new df


        df_results.loc[verb_phrase] = [row['verb-type'],
                                       row['theme-type'],
                                       row['phrase-type'],
                                       row['location-type']
                                       ] + scores

    df_results.to_csv(save_path / 'df_sr.csv')

def predict_next_word(dsm, seq_tok, step_bound):
    seq_set = []
    for seq in seq_tok:
        if seq not in seq_set:
            seq_set.append(seq)

    hit = 0

    if isinstance(dsm, LON) or isinstance(dsm, CTN):
        for seq in seq_set:
            sorted_activation = dsm.spreading_activation(seq[:-1], step_bound)
            rank = [k for k,v in sorted_activation.items()]
            print(seq, rank)
            if rank[0] == seq[-1]:
                hit = hit + 1
        hit = hit/len(seq_set)

    print(hit)









