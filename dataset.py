import pandas as pd
import torch
from torch.utils.data import DataLoader

# The bigger, the better. Depends on your GPU capabilities.
BATCH_SIZE = 16
MAX_SENTENCE_LENGTH = 256

class SentimentDataset:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _rpad(array, n):
        current_len = len(array)
        if current_len > n:
            return array[:n]
        extra = n - current_len
        return array + ([0] * extra)

    def convert_to_embedding(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:MAX_SENTENCE_LENGTH - 2]
        bert_sent = self._rpad(self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]),
                               n=MAX_SENTENCE_LENGTH)

        return bert_sent

    def convert_data_to_embeddings(self, sentences_with_labels):
        label_dict = {'pos': 0, 'neg': 1, 'neu': 2}
        for sentence, label in sentences_with_labels:
            bert_sent = self.convert_to_embedding(sentence)
            label_id = label_dict[label]
            yield torch.tensor(bert_sent), torch.tensor(label_id, dtype=torch.int64)


    @staticmethod
    def _read_csv_data(filename):
        df = pd.read_csv(filename)
        data = df['cleaned_comment'].tolist()
        labels = df['polarity'].map({'pos': 1, 'neg': 0, 'neu': 2}).tolist()
        return list(zip(data, labels))

    def prepare_dataloader_from_examples(self, examples, sampler=None):
        dataset = list(self.convert_data_to_embeddings(examples))

        sampler_func = sampler(dataset) if sampler is not None else None
        dataloader = DataLoader(dataset, sampler=sampler_func, batch_size=BATCH_SIZE)

        return dataloader

    def prepare_dataloader(self, filename, sampler=None):
        data = self._read_csv_data(filename)
        return self.prepare_dataloader_from_examples(data, sampler=sampler)
