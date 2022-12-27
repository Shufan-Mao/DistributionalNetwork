import nltk
from missingadjunct.corpus import Corpus

corpus = Corpus(include_location=True,
                include_location_specific_agents=False,
                seed=1,
                num_epochs=1,
                )

for sentence in corpus.get_sentences():
    print(sentence)



