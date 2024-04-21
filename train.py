import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartTokenizer
from transformers import ViTImageProcessor

from dataset import MMSSDataset
from model import BartForMMSS
from scorer import rouge_score


def post_process(hypothesis):
    hypothesis = hypothesis.replace('<s>', '')
    hypothesis = hypothesis.replace('<pad>', '')
    hypothesis = hypothesis.replace('</s>', '')
    hypothesis = hypothesis.replace("'s", " 's")
    hypothesis = hypothesis.strip()
    return hypothesis


def train(args, train_batch, val_batch, model, tokenizer):
    no_weight_decay = ['bias', 'LayerNorm.weight']
    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters() if
                       not any(nwd in name for nwd in no_weight_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [param for name, param in model.named_parameters() if
                       any(nwd in name for nwd in no_weight_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(
        params=parameter_groups,
        lr=args.learning_rate
    )

    val_rouge_history = [-1]

    for epoch in range(1, args.num_epochs + 1):
        args.logger.info('------Epoch: {epoch:d}------'.format(epoch=epoch))
        train_total_loss = 0
        train_masked_lm_loss = 0
        train_kld_loss = 0
        train_positive_loss = 0
        train_negative_loss = 0
        for step, inputs in tqdm(enumerate(train_batch)):
            inputs = (t.to(args.device) for t in inputs)
            pixel_values, sentence_input_ids, sentence_attention_mask, summarization_input_ids, summarization_attention_mask, labels = inputs

            model.train()
            losses = model(pixel_values=pixel_values,
                           sentence_input_ids=sentence_input_ids,
                           sentence_attention_mask=sentence_attention_mask,
                           summarization_input_ids=summarization_input_ids,
                           summarization_attention_mask=summarization_attention_mask,
                           labels=labels)

            total_loss, masked_lm_loss, kld_loss, positive_loss, negative_loss = losses

            total_loss = total_loss.sum()
            train_total_loss += total_loss.item()

            train_masked_lm_loss += masked_lm_loss.sum().item()
            train_kld_loss += kld_loss.sum().item()
            train_positive_loss += positive_loss.sum().item()
            train_negative_loss += negative_loss.sum().item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        args.logger.info('Total Loss: {total_loss:<10.5f}'.format(
            total_loss=train_total_loss / len(train_batch)))
        args.logger.info('Masked LM Loss: {masked_lm_loss:<10.5f}'.format(
            masked_lm_loss=train_masked_lm_loss / len(train_batch)))
        args.logger.info('KLD Loss: {kld_loss:<10.5f}'.format(
            kld_loss=train_kld_loss / len(train_batch)))
        args.logger.info('Positive Loss: {positive_loss:<10.5f}'.format(
            positive_loss=train_positive_loss / len(train_batch)))
        args.logger.info('Negative Loss: {negative_loss:<10.5f}'.format(
            negative_loss=train_negative_loss / len(train_batch)))

        if args.eval:
            args.logger.info('---evaluation---')

            val_rouge_score = evaluate(args=args, val_batch=val_batch, model=model.module, tokenizer=tokenizer)
            val_rouge_1 = val_rouge_score['rouge-1']['f']
            val_rouge_2 = val_rouge_score['rouge-2']['f']
            val_rouge_l = val_rouge_score['rouge-l']['f']
            args.logger.info(
                'Val Rouge_1: {rouge_1:<10.5f} Rouge_2: {rouge_2:<10.5f} Rouge_l: {rouge_l:<10.5f}'.format(
                    rouge_1=val_rouge_1, rouge_2=val_rouge_2, rouge_l=val_rouge_l))

            if val_rouge_2 > max(val_rouge_history):
                args.logger.info('Best Model!')
                torch.save(obj=model.module, f='best_model.pt')
            val_rouge_history.append(val_rouge_2)


@torch.no_grad()
def evaluate(args, val_batch, model, tokenizer):
    hypotheses = []
    for step, inputs in tqdm(enumerate(val_batch)):
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

    references = []
    val_texts_file_path = os.path.join(args.dataset_path, 'val', 'texts.json')
    with open(file=val_texts_file_path, mode='r', encoding='utf-8') as val_texts_file:
        val_texts = json.load(fp=val_texts_file)

    val_texts = val_texts[:round(args.val_dataset_proportion * len(val_texts))]

    for vt in val_texts:
        references.append(vt['summarization'])

    return rouge_score(hypotheses=post_process_hypotheses, references=references)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--dataset_path', type=str, default='./dataset_mmss')
    parser.add_argument('--vision_foundation_model', type=str, default='google-vit-base-patch32-224-in21k')
    parser.add_argument('--language_foundation_model', type=str, default='facebook-bart-base')

    parser.add_argument('--max_sentence_length', type=int, default=55)
    parser.add_argument('--max_summarization_length', type=int, default=30)

    parser.add_argument('--train_dataset_proportion', type=float, default=0.00025806)
    parser.add_argument('--val_dataset_proportion', type=float, default=0.008)

    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--evaluate_batch_size', type=int, default=16)

    parser.add_argument('--kld_loss_weight', type=float, default=0.01)
    parser.add_argument('--positive_loss_weight', type=float, default=1)
    parser.add_argument('--negative_loss_weight', type=float, default=1)

    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    parser.add_argument('--num_beams', type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename='train.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    args.logger = logger
    args.device = torch.device('cuda')

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    image_processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path=args.vision_foundation_model)
    tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=args.language_foundation_model)

    train_dataset_path = os.path.join(args.dataset_path, 'train')
    train_dataset = MMSSDataset(dataset_path=train_dataset_path, image_processor=image_processor, tokenizer=tokenizer,
                                max_sentence_length=args.max_sentence_length,
                                max_summarization_length=args.max_summarization_length,
                                proportion=args.train_dataset_proportion)
    train_batch = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)

    val_dataset_path = os.path.join(args.dataset_path, 'val')
    val_dataset = MMSSDataset(dataset_path=val_dataset_path, image_processor=image_processor, tokenizer=tokenizer,
                              max_sentence_length=args.max_sentence_length,
                              max_summarization_length=args.max_summarization_length,
                              proportion=args.val_dataset_proportion)
    val_batch = DataLoader(dataset=val_dataset, batch_size=args.evaluate_batch_size, shuffle=False)

    model = BartForMMSS(config=args)

    model = model.to(device=args.device)
    model = torch.nn.DataParallel(module=model)

    train(args=args, train_batch=train_batch, val_batch=val_batch, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()
