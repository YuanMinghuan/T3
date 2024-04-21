import json
import os

from PIL import Image
from torch.utils.data import Dataset


class MMSSDataset(Dataset):

    def __init__(self, dataset_path, image_processor, tokenizer, max_sentence_length, max_summarization_length,
                 proportion):
        texts_file_path = os.path.join(dataset_path, 'texts.json')
        with open(file=texts_file_path, mode='r', encoding='utf-8') as texts_file:
            self.texts = json.load(fp=texts_file)

        self.texts = self.texts[:round(proportion * len(self.texts))]

        self.processed = []

        for text in self.texts:
            # image
            image_file_path = os.path.join(dataset_path, 'images', text['image'])
            image = Image.open(image_file_path).convert('RGB')
            pixel_values = image_processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)

            # text
            sentence = tokenizer(text=text['sentence'], max_length=max_sentence_length,
                                 padding='max_length', truncation=True, return_tensors='pt')

            sentence_input_ids = sentence['input_ids'].squeeze(0)
            sentence_attention_mask = sentence['attention_mask'].squeeze(0)

            summarization = tokenizer(text_target=text['summarization'], max_length=max_summarization_length,
                                      padding='max_length', truncation=True, return_tensors='pt')

            summarization_input_ids = summarization['input_ids'].squeeze(0)
            summarization_attention_mask = summarization['attention_mask'].squeeze(0)

            labels = summarization_input_ids.masked_fill(summarization_input_ids == tokenizer.pad_token_id,
                                                         -100).squeeze(0)

            self.processed.append(
                (
                    pixel_values,
                    sentence_input_ids,
                    sentence_attention_mask,
                    summarization_input_ids,
                    summarization_attention_mask,
                    labels
                )
            )

    def __getitem__(self, index):
        return self.processed[index]

    def __len__(self):
        return len(self.processed)
