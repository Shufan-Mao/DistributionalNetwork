from pathlib import Path
from typing import List, Tuple
import pandas as pd
import random
import numpy as np

from src.MissingAdjunct.missingadjunct.corpus import Corpus
from src.MissingAdjunct.missingadjunct.utils import make_blank_sr_df


from src.params import Params
from src.other_dsms.count import CountDSM
from src.other_dsms.w2vec import W2Vec
from src.other_dsms.rnn import RNN
from src.other_dsms.transformer import Transformer
from src.networks.ctn import CTN
from src.networks.lon import LON

from src.tasks import select_instrument, predict_next_word

p2val = {'dsm':'ctn',
         'save_path':'Data',
         'excluded_tokens':None,
         'include_location':False,
        'include_location_specific_agents':False,
        'num_blocks':400,
        'complete_block':True,
        'add_with':False,
        'add_in':True,
        'strict_compositional':False,
        'add_reversed_seq':False,
        'composition_fn': 'native'

         }
decay = 0.75
step_bound = 2 # non-recurrent activation if None, recurrent activation with bound if certain number


def main(param2val):
    """
    Train a single DSM once, and save results
    """

    # params
    params = Params.from_param2val(param2val)
    print(params)

    save_path = Path(param2val['save_path'])

    # in case job is run locally, we must create save_path
    if not save_path.exists():
        save_path.mkdir(parents=True)


    corpus = Corpus(include_location=params.corpus_params.include_location,
                    include_location_specific_agents=params.corpus_params.include_location_specific_agents,
                    num_epochs=params.corpus_params.num_blocks,
                    complete_epoch=params.corpus_params.complete_block,
                    seed=random.randint(0, 1000),
                    add_with=params.corpus_params.add_with,
                    add_in=params.corpus_params.add_in,
                    strict_compositional=params.corpus_params.strict_compositional,
                    )

    pepper_count = 0
    orange_count = 0

    for sentence in corpus.get_sentences():
        if 'preserve orange' in sentence:
            orange_count = orange_count + 1

        elif 'preserve pepper' in sentence:
            pepper_count = pepper_count + 1

    print(pepper_count, orange_count)


    # load blank df for evaluating sr scores
    df_blank = make_blank_sr_df()
    #df_blank.insert(loc=3, column='location-type', value=['' for i in range(df_blank.shape[0])])
    instruments = df_blank.columns[4:]  # instrument columns start after the 4th column
    if not set(instruments).issubset(corpus.vocab):
        raise RuntimeError('Not all instruments in corpus. Add more blocks or set complete_block=True')

    # collect corpus data
    seq_num: List[List[int]] = []  # sequences of Ids
    seq_tok: List[List[str]] = []  # sequences of tokens
    seq_parsed: List[Tuple] = []  # sequences that are constituent-parsed

    # TODO: in newer version of MissingAdjunct, token2id, eos are attributes of the Corpus class
    #token2id = {t: n for n, t in enumerate(corpus.vocab)}
    #eos = '<eos>'

    for s in corpus.get_sentences():  # a sentence is a string
        tokens = s.split()
        seq_num.append([corpus.token2id[token] for token in tokens])  # numeric (token IDs)
        seq_tok.append(tokens)  # raw tokens
        if params.corpus_params.add_reversed_seq:
            seq_num.append([corpus.token2id[token] for token in tokens][::-1])
            seq_tok.append(tokens[::-1])
    for tree in corpus.get_trees():
        seq_parsed.append(tree)

    print(f'Number of sequences in corpus={len(seq_tok):,}', flush=True)

    if params.dsm == 'count':
        dsm = CountDSM(params.dsm_params, corpus.vocab, seq_num)
    elif params.dsm == 'w2v':
        dsm = W2Vec(params.dsm_params, corpus.vocab, seq_tok)
    elif params.dsm == 'rnn':
        dsm = RNN(params.dsm_params, corpus.token2id, seq_num, df_blank, instruments, save_path)
    elif params.dsm == 'transformer':
        dsm = Transformer(params.dsm_params, corpus.token2id, seq_num, df_blank, instruments, save_path, corpus.eos)
    elif params.dsm == 'ctn':
        dsm = CTN(params.dsm_params, corpus.token2id, seq_parsed, decay)

    elif params.dsm == 'lon':
        dsm = LON(params.dsm_params, seq_tok, decay)  # TODO the net is built directly from corpus rather than co-occ
    else:
        raise NotImplementedError


    # Train distributional models
    dsm.train()
    print(f'Completed training the DSM', flush=True)

    if params.dsm == 'ctn' or params.dsm == 'lon':
        dsm.get_accumulated_activations(step_bound)

    ####################################################################################################################
    #Tasks
    ####################################################################################################################


    # verb phrase select instrument
    #select_instrument(df_blank, dsm, instruments, params, step_bound, save_path)

    # predicting next word
    predict_next_word(dsm, seq_tok, step_bound)


    ####################################################################################################################
    ####################################################################################################################


    # prepare collected data for returning to Ludwig
    performance = dsm.get_performance()
    series_list = []
    for k, v in performance.items():
        if k == 'epoch':
            continue
        s = pd.Series(v, index=performance['epoch'])
        s.name = k
        series_list.append(s)

    # save model
    if isinstance(dsm, Transformer):
        dsm.model.save_pretrained(str(save_path))

    print('Completed main.job.', flush=True)

    return series_list

main(p2val)