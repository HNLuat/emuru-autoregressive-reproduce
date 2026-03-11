from typing import Optional, Union
from pathlib import Path
import json
import os
import pandas as pd

from cv2 import transform
import torch
import webdataset as wds
from einops import rearrange
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from hwd.datasets.shtg import KaraokeLines
import torchvision.transforms.functional as TF
from torch.utils.data import Sampler
import random


from .alphabet import Alphabet
from .constants import FONT_SQUARE_CHARSET
from .subsequent_mask import subsequent_mask


class DataProcessor:
    """Handles data loading and processing operations"""
    
    @staticmethod
    def pad_images(images, padding_value=1):
        """Pad images to same width for batching"""
        images = [rearrange(img, 'c h w -> w c h') for img in images]
        return rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')
    
    @staticmethod
    def pad_images_fixed(images, max_width=768, padding_value=1):
        """Pad images to fixed maximum width"""
        padded_images = []
        for img in images:
            c, h, w = img.shape
            if w > max_width:
                img = img[:, :, :max_width]  # Crop if too wide
                w = max_width
            
            if w < max_width:
                pad_width = max_width - w
                img = torch.nn.functional.pad(img, (0, pad_width), value=padding_value)
            
            padded_images.append(img)
        
        return torch.stack(padded_images)


class WIDCollate:
    def __call__(self, batch):
        """Collate function for HuggingFace webdataset format"""
        bw_imgs = [sample['bw.png'] for sample in batch]
        bw_imgs_padded = DataProcessor.pad_images_fixed(bw_imgs)
        
        writer_ids = [sample['json']['writer_id'] for sample in batch]

        return {
            'bw': bw_imgs_padded,
            'writer_id': torch.tensor(writer_ids),
        }   
    

class VAECollate:
    def __init__(self, alphabet):
        self.alphabet = alphabet
    
    def __call__(self, batch):
        """Collate function for HuggingFace webdataset format"""
        rgb_imgs = [sample['rgb.png'] for sample in batch]
        bw_imgs = [sample['bw.png'] for sample in batch]
        rgb_imgs_padded = DataProcessor.pad_images_fixed(rgb_imgs)
        bw_imgs_padded = DataProcessor.pad_images_fixed(bw_imgs)
        
        writer_ids = [sample['json']['writer_id'] for sample in batch]

        gen_texts = [sample['json']['text'] for sample in batch]
        texts_len = [len(gen_text) for gen_text in gen_texts]
        encoded_texts = [torch.LongTensor(sample['encoded_text']) for sample in batch]
        text_logits_s2s = [torch.cat([torch.LongTensor([self.alphabet.sos]), encoded_text, torch.LongTensor([self.alphabet.eos])]) for encoded_text in encoded_texts]
        
        text_logits_s2s = pad_sequence(text_logits_s2s, batch_first=True, padding_value=self.alphabet.pad)
        texts_len = torch.LongTensor(texts_len)
        tgt_key_mask = subsequent_mask(text_logits_s2s.shape[-1] - 1)
        tgt_key_padding_mask = text_logits_s2s == self.alphabet.pad

        return {
            'rgb': rgb_imgs_padded,
            'bw': bw_imgs_padded,
            'writer_id': torch.tensor(writer_ids),
            'text_logits_s2s': text_logits_s2s,
            'texts_len': texts_len,
            'tgt_key_mask': tgt_key_mask,
            'tgt_key_padding_mask': tgt_key_padding_mask,
        }   


class T5Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        txts = [sample['json']['text'] for sample in batch]
        res = self.tokenizer(txts, padding=True, return_tensors='pt', return_attention_mask=True, return_length=True)
        res['img'] = DataProcessor.pad_images([sample['rgb.png'] for sample in batch])
        res['text'] = txts
        return res



class SHTGWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.transforms = T.Compose([
            self._to_height_64,
            T.ToTensor()
        ])

    def _to_height_64(self, img):
        width, height = img.size
        aspect_ratio = width / height
        new_width = int(64 * aspect_ratio)
        resized_image = img.resize((new_width, 64), Image.LANCZOS)
        return resized_image

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        sample['style_img'] = self.transforms(sample['style_imgs'][0].convert('RGB'))
        return sample


def karaoke_collate_fn(batch):
    out = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        # Stack images/tensors
        if key in ['style_img', 'gen_img']:
            tensorized = []
            for v in values:
                if isinstance(v, torch.Tensor):
                    tensorized.append(v)
                else:  # Assume PIL Image
                    tensorized.append(transforms.ToTensor()(v))
            out[key] = torch.stack(tensorized)
        elif isinstance(values[0], (str, int, float)):
            out[key] = values
        elif isinstance(values[0], (list, tuple)):
            if values[0] and isinstance(values[0][0], Path):
                out[key] = [[str(p) for p in v] for v in values]
            else:
                out[key] = values
        elif isinstance(values[0], Path):
            out[key] = [str(v) for v in values]
        else:
            out[key] = values
    return out

class WidthBucketSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_size=100, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        
        # lấy width của tất cả sample
        self.indices = list(range(len(dataset)))
        self.indices.sort(key=lambda i: dataset[i]['width'])

    def __iter__(self):
        buckets = [
            self.indices[i:i+self.bucket_size]
            for i in range(0, len(self.indices), self.bucket_size)
        ]

        batches = []
        
        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.indices) // self.batch_size

class IAMWordDataset(Dataset):
    def __init__(self, root, csv_path, split, alphabet, transform):
        self.root = root
        self.alphabet = alphabet
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.root, row["image_path"])
        img = Image.open(img_path)

        rgb = img.convert("RGB")
        bw = img.convert("L")

        text = str(row["label"])

        sample = {
            "rgb.png": self.transform(rgb),
            "bw.png": self.transform(bw),
            "json": {
                "text": text,
                "writer_id": row.get("writer_id", -1),
            },
            "encoded_text": self.alphabet.encode(text),
            "width": rgb.size[0],
        }

        return sample

def resize_keep_ratio_and_pad(
    img: Image.Image,
    target_h: int,
):
    w, h = img.size

    # resize theo height
    new_w = int(w * target_h / h)
    img = img.resize((new_w, target_h), Image.BILINEAR)

    return img

def iam_transform(img: Image.Image):

    img = resize_keep_ratio_and_pad(
        img,
        target_h=64,
    )

    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.5], [0.5])(img)

    return img

class DataLoaderManager:
    """Handles dataset creation and data loading"""
    
    def __init__(self, train_pattern: str, eval_pattern: str, train_batch_size: int, eval_batch_size: int, num_workers: int, pin_memory: bool, 
        persistent_workers: bool, tokenizer: Optional[AutoTokenizer] = None):

        self.train_pattern = train_pattern
        self.eval_pattern = eval_pattern
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.alphabet = Alphabet(FONT_SQUARE_CHARSET)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        
    def create_dataset(self, split: str, model_type: str):
        """Create training dataset using WebDataset"""

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        collate_fn: Union[VAECollate, WIDCollate, T5Collate]
        if model_type == 'vae' or model_type == 'htr':
            collate_fn = VAECollate(self.alphabet)
        elif model_type == 'wid':
            collate_fn = WIDCollate()
        elif model_type == 't5':
            collate_fn = T5Collate(self.tokenizer)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        if split == 'train':
            pattern = self.train_pattern
        elif split == 'eval':
            pattern = self.eval_pattern
        else:
            raise ValueError(f"Invalid split: {split}")
         
        shuffle = 100 if split == 'train' else 0
        dataset = (
            wds.WebDataset(pattern,  nodesplitter=wds.split_by_node, shardshuffle=shuffle)
            .decode("pil")
            .map(lambda sample: {
                "rgb.png": transform(sample["rgb.png"].convert('RGB')),
                "bw.png": transform(sample["bw.png"].convert('L')),
                'json': sample['json'],
                'encoded_text': self.alphabet.encode(sample['json']['text']),
            })
        )
        
        return DataLoader(
            dataset, 
            batch_size=self.train_batch_size if split == 'train' else self.eval_batch_size,
            pin_memory=self.pin_memory if split == 'train' else False,
            collate_fn=collate_fn,
            num_workers=self.num_workers if split == 'train' else 0,
            drop_last=True if split == 'train' else False,
            persistent_workers=self.persistent_workers if split == 'train' else False
        )

    def create_karaoke_dataset(self):
        """Create dataset with Karaoke datasets"""
        karaoke_handw = KaraokeLines('handwritten', num_style_samples=1, load_gen_sample=True)
        karaoke_typew = KaraokeLines('typewritten', num_style_samples=1, load_gen_sample=True)
        eval_dataset = ConcatDataset([SHTGWrapper(karaoke_handw), SHTGWrapper(karaoke_typew)])

        return DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=karaoke_collate_fn,
            num_workers=self.num_workers,
            persistent_workers=False
        )

    def create_iam_dataset(
        self,
        root: str,
        label_csv: str,
        model_type: str,
    ):
        if model_type == 'vae' or model_type == 'htr':
            collate_fn = VAECollate(self.alphabet)
        elif model_type == 'wid':
            collate_fn = WIDCollate()
        elif model_type == 't5':
            collate_fn = T5Collate(self.tokenizer)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        train_set = IAMWordDataset(
            root=root,
            csv_path=label_csv,
            split="train",
            alphabet=self.alphabet,
            transform=iam_transform,
        )

        val_set = IAMWordDataset(
            root=root,
            csv_path=label_csv,
            split="eval",
            alphabet=self.alphabet,
            transform=iam_transform,
        )

        # train_sampler = WidthBucketSampler(train_set, batch_size=self.train_batch_size)

        val_sampler = WidthBucketSampler(val_set, batch_size=self.eval_batch_size)

        # train_loader = DataLoader(
        #     dataset=train_set,
        #     batch_sampler=train_sampler,
        #     num_workers=self.num_workers,
        #     pin_memory=self.pin_memory,
        #     collate_fn=collate_fn,
        #     persistent_workers=self.persistent_workers,
        # )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

        val_loader = DataLoader(
            dataset=val_set,
            batch_sampler=val_sampler,
            num_workers=4,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader
