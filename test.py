import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartTokenizer
from transformers import ViTImageProcessor

from dataset import MMSSDataset


def post_process(hypothesis):
    hypothesis = hypothesis.replace('<s>', '')
    hypothesis = hypothesis.replace('<pad>', '')
    hypothesis = hypothesis.replace('</s>', '')
    hypothesis = hypothesis.replace("'s", " 's")
    hypothesis = hypothesis.strip()
    return hypothesis


@torch.no_grad()
def test(args, test_batch, model, tokenizer):
    hypotheses = []
    for step, inputs in tqdm(enumerate(test_batch)):
        inputs = (t.to(args.device) for t in inputs)
        pixel_values, sentence_input_ids, sentence_attention_mask, summarization_input_ids, summarization_attention_mask, labels = inputs

        model.eval()
        generated_ids = model.generate(pixel_values=pixel_values,
                                       sentence_input_ids=sentence_input_ids,
                                       sentence_attention_mask=sentence_attention_mask)

        generated_summarization = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        hypotheses += generated_summarization

    post_process_hypotheses = []
    for hyp in hypotheses:
        post_process_hypotheses.append(post_process(hyp))

    test_texts_file_path = os.path.join(args.dataset_path, 'test', 'texts.json')
    with open(file=test_texts_file_path, mode='r', encoding='utf-8') as test_texts_file:
        test_texts = json.load(fp=test_texts_file)

    test_texts = test_texts[:round(args.test_dataset_proportion * len(test_texts))]

    for tt, pph in zip(test_texts, post_process_hypotheses):
        tt['generated summarization'] = pph

    return test_texts


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--dataset_path', type=str, default='./dataset_mmss')
    parser.add_argument('--vision_foundation_model', type=str, default='google-vit-base-patch32-224-in21k')
    parser.add_argument('--language_foundation_model', type=str, default='facebook-bart-base')

    parser.add_argument('--max_sentence_length', type=int, default=55)
    parser.add_argument('--max_summarization_length', type=int, default=30)

    parser.add_argument('--test_dataset_proportion', type=float, default=1.0)

    parser.add_argument('--evaluate_batch_size', type=int, default=16)

    parser.add_argument('--num_beams', type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    args.device = torch.device('cuda')

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    image_processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path=args.vision_foundation_model)
    tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=args.language_foundation_model)

    test_dataset_path = os.path.join(args.dataset_path, 'test')
    test_dataset = MMSSDataset(dataset_path=test_dataset_path, image_processor=image_processor, tokenizer=tokenizer,
                               max_sentence_length=args.max_sentence_length,
                               max_summarization_length=args.max_summarization_length,
                               proportion=args.test_dataset_proportion)
    test_batch = DataLoader(dataset=test_dataset, batch_size=args.evaluate_batch_size, shuffle=False)

    best_model_path = 'best_model.pt'
    best_model = torch.load(f=best_model_path)

    test_texts = test(args=args, test_batch=test_batch, model=best_model, tokenizer=tokenizer)

    write_file = open(file='./test.json', mode='w', encoding='utf-8')
    json.dump(obj=test_texts, fp=write_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
