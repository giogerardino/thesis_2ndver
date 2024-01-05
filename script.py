import os
import argparse
import pandas as pd

from torch.utils.data import RandomSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split

from dataset import SentimentDataset
from model import SentimentBERT

BERT_MODEL = 'bert-base-uncased'
NUM_LABELS = 3  # positive, negative, and neutral reviews

parser = argparse.ArgumentParser(prog='script')
parser.add_argument('--train', action="store_true", help="Train new weights")
parser.add_argument('--evaluate', action="store_true", help="Evaluate existing weights")
parser.add_argument('--predict', default="", type=str, help="Predict sentiment on a given sentence")
parser.add_argument('--path', default='weights/', type=str, help="Weights path")
parser.add_argument('--train-file', default='manually_labeled.csv',
                    type=str, help="Your train file. One sentence per line.")
parser.add_argument('--test-file', default='manually_labeled.csv',
                    type=str, help="Your test file. One sentence per line.")
args = parser.parse_args()

def train(train_file, epochs=20, output_dir="weights/"):
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)

    dt = SentimentDataset(tokenizer)
    df = pd.read_csv(train_file)
    train_df, _ = train_test_split(df, test_size=0.2)
    dataloader = dt.prepare_dataloader_from_examples(zip(train_df['cleaned_comment'], train_df['polarity']), sampler=RandomSampler)
    predictor = SentimentBERT()
    predictor.train(tokenizer, dataloader, model, epochs)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def evaluate(test_file, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)

    dt = SentimentDataset(predictor.tokenizer)
    df = pd.read_csv(test_file)
    _, val_df = train_test_split(df, test_size=0.2)
    dataloader = dt.prepare_dataloader_from_examples(zip(val_df['cleaned_comment'], val_df['polarity']))
    score = predictor.evaluate(dataloader, val_df['polarity'])
    print(score)

def predict(text, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)

    dt = SentimentDataset(predictor.tokenizer)
    dataloader = dt.prepare_dataloader_from_examples([(text, 'neu')], sampler=None)   # text and a dummy label
    result = predictor.predict(dataloader)

    return "Positive" if result[0] == 0 else ("Negative" if result[0] == 1 else "Neutral")

if __name__ == '__main__':
    if args.train:
        os.makedirs(args.path, exist_ok=True)
        train(args.train_file, epochs=10, output_dir=args.path)

    if args.evaluate:
        evaluate(args.test_file, model_dir=args.path)

    if len(args.predict) > 0:
        print(predict(args.predict, model_dir=args.path))
