
from missingadjunct.corpus import Corpus



corpus = Corpus(include_location=True,
                include_location_specific_agents=False,
                seed=1,
                num_epochs=1,
                #complete_epoch=True,
                #add_with=True,
                #add_in=False,
                #strict_compositional=True,  # todo test
                )

for sentence in corpus.get_sentences():
    if 'bake' in sentence:
        print(sentence)
