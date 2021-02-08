from typing import List
import torch
from gensim.models import word2vec


class Preprocessor:

    def __init__(self, corpus: List[List[str]]) -> None:
        # skip-gram
        corpus = corpus + [5 * ['<PAD>'], 5 * ['<UNK>']]
        self.embedding = word2vec.Word2Vec(corpus, size=250, window=5, min_count=5, workers=4, iter=10, sg=1)
        self.sentence_len = 20

    def _padding(self, sentence) -> List[int]:
        if len(sentence) > self.sentence_len:
            return sentence[:self.sentence_len]
        else:
            sentence.extend((self.sentence_len - len(sentence)) * [self._word_to_idx('<PAD>')])
            return sentence

    def _word_to_idx(self, word: str) -> int:
        return self.embedding.wv.vocab[word].index

    def sentence_to_vector(self, sentence: List[str]) -> torch.LongTensor:
        vector = []
        word_unk = self._word_to_idx('<UNK>')
        for word in sentence:
            if word in self.embedding.wv.vocab:
                vector.append(self._word_to_idx(word))
            else:
                vector.append(word_unk)
        return torch.LongTensor(self._padding(vector))

    def labels_to_tensor(self, labels: List[int]) -> torch.LongTensor:
        return torch.LongTensor([label for label in labels])

    def get_embedding(self):
        return torch.FloatTensor(self.embedding.wv.vectors)