import torch
from torch.utils.data import Dataset
import spacy
from collections import Counter

class Vocabulary:
    def __init__(self, texts, specials, min_freq=1):
        self.specials = specials
        counter = Counter(token for text in texts for token in text)
        self.itos = list(specials)
        for token, freq in counter.items():
            if freq >= min_freq and token not in specials:
                self.itos.append(token)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.default_index = self.stoi["<unk>"]

    def __call__(self, tokens):
        return [self.stoi.get(tok, self.default_index) for tok in tokens]

    def set_default_index(self, idx):
        self.default_index = idx

    def lookup_tokens(self, indices):
        return [self.itos[i] for i in indices]
    
    def __len__(self):
        return len(self.itos)


class ParallelCorpus(Dataset):
    def __init__(self, data, lang_a, lang_b, vocab_a=None, vocab_b=None, data_limit=torch.inf):

        self.nlp_a = spacy.blank(lang_a)
        self.nlp_b = spacy.blank(lang_b)

        self.texts_a, self.texts_b = [], []

        for src, tgt in data:
            tokens_a = [t.text for t in self.nlp_a(src.strip())]
            tokens_b = [t.text for t in self.nlp_b(tgt.strip())]

            tokens_a = ["<start>"] + tokens_a + ["<stop>"]
            tokens_b = ["<start>"] + tokens_b + ["<stop>"]

            self.texts_a.append(tokens_a)
            self.texts_b.append(tokens_b)

            if len(self.texts_a) >= data_limit:
                break

        specials = ["<pad>", "<unk>", "<start>", "<stop>"]

        if vocab_a is None or vocab_b is None:
            self.vocab_a = Vocabulary(self.texts_a, specials, min_freq=2)
            self.vocab_b = Vocabulary(self.texts_b, specials, min_freq=2)
        else:
            self.vocab_a, self.vocab_b = vocab_a, vocab_b

        self.texts_a = [
            torch.tensor(self.vocab_a(text), dtype=torch.int64)
            for text in self.texts_a
        ]
        self.texts_b = [
            torch.tensor(self.vocab_b(text), dtype=torch.int64)
            for text in self.texts_b
        ]

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, idx):
        return {
            "text_a": self.texts_a[idx],
            "text_b": self.texts_b[idx],
        }

    def convert_to_text(self, text_a, text_b):
        return (
            self.vocab_a.lookup_tokens(text_a),
            self.vocab_b.lookup_tokens(text_b),
        )


def translate(sentence, model, dataset, device, max_len=50, verbose=True):
    model.eval()

    src_tensor = torch.LongTensor(sentence).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(sentence)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    trg_indexes = dataset.vocab_b(["<start>"])
    stop_token = dataset.vocab_b(["<stop>"])[0]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(
                trg_tensor, hidden, encoder_outputs, src_len
            )

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == stop_token:
            break

    trg_tokens = dataset.vocab_b.lookup_tokens(trg_indexes)

    if verbose:
        print(f"Źródło = {dataset.vocab_a.lookup_tokens(sentence.numpy())}")
        print(f"Tłumaczenie = {trg_tokens}")

    return trg_tokens