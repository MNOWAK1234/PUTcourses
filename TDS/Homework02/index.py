from datetime import datetime
from itertools import chain

import weaviate
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

client = weaviate.Client("http://localhost:8080")

def words_from_synset(synset: Synset):
    return [l.name().replace("_", " ") for l in synset.lemmas()]

schema = {
    "classes": [
        {
            "class": "Phrase",
            "description": "Fraza z WordNet",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "Treść frazy"
                },
                {
                    "name": "word_count",
                    "dataType": ["int"],
                    "description": "Liczba słów w frazie"
                }
            ]
        }
    ]
}

client.schema.delete_all()
client.schema.create(schema)

class RequestCounter:
    count: int = 0
    time: datetime = datetime.now()

    def __call__(self, results):
        self.count += len(results)
        print(
            f"{len(results):6} created | {self.count:6} total | time: {datetime.now() - self.time}"
        )

with client.batch(batch_size=2048, callback=RequestCounter(), timeout_retries=8) as batch:

    wordnet_iter = (
        lemma
        for lemma in chain.from_iterable(
            words_from_synset(synset)
            for synset in wn.all_synsets("n")
        )
    )

    for i, phrase in enumerate(wordnet_iter):
        if i >= 10_000:
            break

        batch.add_data_object(
            data_object={
                "text": phrase,
                "word_count": len(phrase.split())
            },
            class_name="Phrase"
        )