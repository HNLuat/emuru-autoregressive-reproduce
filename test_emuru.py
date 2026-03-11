import argparse
from pathlib import Path
import math
import uuid
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import csv
import pandas as pd

import torch
from torchvision.models import inception_v3
import torch.nn as nn
import wandb
from PIL import Image
from torchvision.transforms import functional as F
from loguru import logger
from accelerate import Accelerator
from accelerate.utils import broadcast
from accelerate.utils import ProjectConfiguration, set_seed
from transformers.optimization import get_scheduler
from torchmetrics.image.kid import KernelInceptionDistance

from utils import TrainState
from custom_datasets import DataLoaderManager
from models.emuru import Emuru, EmuruConfig

from tqdm import tqdm


from scipy import linalg

def compute_fid(real_feats, fake_feats):

    mu1 = real_feats.mean(0)
    mu2 = fake_feats.mean(0)

    sigma1 = np.cov(real_feats, rowvar=False)
    sigma2 = np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(
        sigma1 + sigma2 - 2 * covmean
    )

    return fid

def compute_kid(real_loader, fake_loader, device):

    kid = KernelInceptionDistance(
        feature=2048,
        subset_size=100
    ).to(device)

    for x in real_loader:
        kid.update(x.to(device), real=True)

    for x in fake_loader:
        kid.update(x.to(device), real=False)

    return kid.compute()

def compute_bfid(real_feats, fake_feats, batch=100):

    n = min(len(real_feats), len(fake_feats))

    scores = []

    for i in range(0, n, batch):

        r = real_feats[i:i+batch]
        f = fake_feats[i:i+batch]

        if len(r) < batch:
            break

        scores.append(
            compute_fid(r.numpy(), f.numpy())
        )

    return np.mean(scores)

import editdistance

def compute_cer(preds, gts):
    total_dist = 0
    total_chars = 0

    for p, g in zip(preds, gts):
        total_dist += editdistance.eval(p, g)
        total_chars += len(g)

    return total_dist / total_chars

def run_htr(model, loader, device):
    preds = []
    model.to(device).eval()
    with torch.no_grad():
        for img in loader:
            img = img.to(device)
            txt = model.predict(img)
            preds.extend(txt)

    return preds

class ImageFolderDataset(Dataset):
    def __init__(self, root, size=299):
        self.paths = [os.path.join(root, p) for p in os.listdir(root)]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

class InceptionFeature(nn.Module):

    def __init__(self):
        super().__init__()
        model = inception_v3(pretrained=True)
        model.fc = nn.Identity()
        self.model = model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

def extract_features(model, loader, device):
    feats = []
    model.to(device)

    for x in tqdm(loader):
        x = x.to(device)
        f = model(x)
        feats.append(f.cpu())

    feats = torch.cat(feats)
    return feats

def evaluate_model(
        real_dir,
        fake_dir,
        htr_model,
        gt_texts,
        device="cuda"
):

    real_ds = ImageFolderDataset(real_dir)
    fake_ds = ImageFolderDataset(fake_dir)

    real_loader = DataLoader(real_ds, batch_size=32)
    fake_loader = DataLoader(fake_ds, batch_size=32)

    feat_net = InceptionFeature()

    real_feats = extract_features(feat_net, real_loader, device)
    fake_feats = extract_features(feat_net, fake_loader, device)

    fid = compute_fid(
        real_feats.numpy(),
        fake_feats.numpy()
    )

    bfid = compute_bfid(real_feats, fake_feats)

    kid = compute_kid(real_loader, fake_loader, device)

    preds = run_htr(htr_model, fake_loader, device)

    cer = compute_cer(preds, gt_texts)

    return {
        "FID": fid,
        "BFID": bfid,
        "KID": kid,
        "CER": cer
    }

def gen_test_image(
    eval_loader,
    model,
    accelerator,
    weight_dtype,
    total_image=1000,
    text_path=None,
    output_dir="results_emmuru"
):

    model = accelerator.unwrap_model(model)
    model.eval()

    number_of_images = 0
    gt_text = []
    unique_texts = []

    if text_path:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
            unique_texts = list(set(content.splitlines()))

    # print(unique_texts[:10])

    image_path = os.path.join(output_dir, "gen_image")
    os.makedirs(image_path, exist_ok=True)

    for step, batch in enumerate(eval_loader):
        if number_of_images >= total_image:
            break
        
        print(f"Generating images: {number_of_images}/{total_image}", end="\r")
        with torch.no_grad():
            with accelerator.autocast():
                images = batch['img'].to(accelerator.device, dtype=weight_dtype)
                text = batch['text']

        gen_text = list(np.random.choice(
            unique_texts if unique_texts else text,
            size=len(text)
        ))

        try:
            gen_img = model.generate_batch(
                style_texts=text,
                gen_texts=gen_text,
                style_imgs=images,
                lengths=[i.size(-1) for i in images]
            )

            remaining = total_image - number_of_images
            gen_img = gen_img[:remaining]
            gen_text = gen_text[:remaining]

            for img, txt in zip(gen_img, gen_text):

                w, h = img.size
                if w < 5 or h < 5:
                    raise ValueError(f"Empty image {img.size}")

                img_name = f"gen_image_{number_of_images}.png"
                save_path = os.path.join(image_path, img_name)

                img.convert("L").save(save_path)

                gt_text.append(txt)
                number_of_images += 1

        except Exception as e:
            print(f"Error generating images. Error: {e}")
            continue

    del model
    torch.cuda.empty_cache()

    generate_label = {
        "image_path": [f"gen_image/gen_image_{i}.png" for i in range(len(gt_text))],
        "label": gt_text
    }

    gt_output_file = os.path.join(output_dir, "gt_texts.csv")
    pd.DataFrame(generate_label).to_csv(gt_output_file, index=False)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results_emuru', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results_t5', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=4, help="train batch size") 
    parser.add_argument("--eval_batch_size", type=int, default=4, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of train epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="reduce_lr_on_plateau")
    parser.add_argument("--lr_scheduler_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24, help="random seed")
   
    parser.add_argument("--eval_epochs", type=int, default=1, help="eval interval")
    parser.add_argument("--resume_id", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--run_id", type=str, default=uuid.uuid4().hex[:4], help="uuid of the run")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default="iam-handwriting-emuru", help="wandb project name")
    parser.add_argument('--wandb_log_interval_steps', type=int, default=200, help="wandb log interval")

    parser.add_argument("--use_train_bucket", type=bool, default=False, help="whether to bucket train dataset")
    parser.add_argument("--use_val_bucket", type=bool, default=False, help="whether to bucket eval dataset")
    parser.add_argument("--dataset_dir", type=str, default="C:/Users/LENOVO/Documents/Python Project/Handwritting_gen/iam_word_dataset", help="dataset directory")

    parser.add_argument("--T5_path", type=str, default="blowing-up-groundhogs/emuru", help='t5 checkpoint path')
    parser.add_argument("--vae_path", type=str, default="blowing-up-groundhogs/emuru_vae", help='vae checkpoint path')

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--checkpoints_total_limit", type=int, default=2)

    parser.add_argument('--teacher_noise', type=float, default=0.1, help='How much noise add during training')
    parser.add_argument('--training_type', type=str, default='pretrain', help='Pre-training or long lines finetune', choices=['pretrain', 'finetune'])

    args = parser.parse_args()

    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.adam_weight_decay = 0.01

    args.run_name = args.resume_id if args.resume_id else args.run_id
    # args.output_dir = Path(args.output_dir) / args.run_name
    # args.logging_dir = Path(args.logging_dir) / args.run_name
    args.output_dir = Path(args.output_dir)
    args.logging_dir = Path(args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=str(args.output_dir),
        logging_dir=str(args.logging_dir),
        automatic_checkpoint_naming=True,
        total_limit=args.checkpoints_total_limit,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        cpu=False,
    )

    logger.info(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.logging_dir.mkdir(parents=True, exist_ok=True)

    emuru_config = EmuruConfig(
        t5_name_or_path='google-t5/t5-large',
        vae_name_or_path=args.vae_path,
        tokenizer_name_or_path='google/byt5-small',
        slices_per_query=1,
        vae_channels=1
    )
    if args.T5_path:
        try:
            model = Emuru.from_pretrained(args.T5_path, config=emuru_config)
            print(f"Loaded T5 from {args.T5_path}")
        except:
            emuru_config.t5_name_or_path = args.T5_path
            model = Emuru(emuru_config)
    else:
        model = Emuru(emuru_config)
        print(f"No pretrained found")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        scheduler_specific_kwargs={"patience": args.lr_scheduler_patience}
    )

    if args.training_type == 'pretrain':
        train_pattern = ("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/train/{000000..000498}.tar")
        eval_pattern = ("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/train/{000499..000499}.tar")
        NUM_SAMPLES_TRAIN = 8_000 * 499
        NUM_SAMPLES_EVAL = 8_000
    elif args.training_type == 'finetune':
        train_pattern = ("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/fine_tune/{000000..000048}.tar")
        eval_pattern = ("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/fine_tune/{000049..000049}.tar")
        NUM_SAMPLES_TRAIN = 8_000 * 49
        NUM_SAMPLES_EVAL = 8_000
    else:
        raise ValueError(f"Invalid training type: {args.training_type}")

    data_loader = DataLoaderManager(
        train_pattern=None,
        eval_pattern=None,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=False,
        persistent_workers=False,
        tokenizer=model.tokenizer,
    )
    # train_loader = data_loader.create_dataset('train', 't5')
    # eval_loader = data_loader.create_dataset('eval', 't5')
    # karaoke_loader = data_loader.create_karaoke_dataset()
    dataset_dir = args.dataset_dir
    train_loader, eval_loader = data_loader.create_iam_dataset(
        root=dataset_dir,
        label_csv=f"{dataset_dir}/label.csv",
        model_type="t5",   # hoặc 'vae', 'wid'
        use_train_bucket=args.use_train_bucket,
        use_val_bucket=args.use_val_bucket
    )
    NUM_SAMPLES_TRAIN = len(train_loader.dataset)
    NUM_SAMPLES_EVAL = len(eval_loader.dataset)


    LEN_EVAL_LOADER = NUM_SAMPLES_EVAL // args.eval_batch_size

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, eval_loader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        wandb_args = {"wandb": {"entity": args.wandb_entity, "name": args.run_name}}
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.wandb_project_name, tracker_config, wandb_args)

    num_steps_per_epoch = math.ceil(NUM_SAMPLES_TRAIN / (args.train_batch_size * args.gradient_accumulation_steps))
    args.max_train_steps = args.epochs * num_steps_per_epoch
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    args.emuru_params = sum([p.numel() for p in model.parameters()])

    logger.info("***** Running T5 training *****")
    logger.info(f"  Num train samples = {NUM_SAMPLES_TRAIN}. Num steps per epoch = {num_steps_per_epoch}")
    logger.info(f"  Num eval samples = {NUM_SAMPLES_EVAL}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total trainable parameters count = {args.emuru_params}")

    train_state = TrainState(global_step=0, epoch=0, best_eval_init=float('inf'))
    accelerator.register_for_checkpointing(train_state)
    if args.resume_id:
        try:
            accelerator.load_state()
            accelerator.project_configuration.iteration = train_state.epoch
            logger.info(f"  Resuming from checkpoint at epoch {train_state.epoch}")
        except FileNotFoundError as e:
            logger.warning(f"  Checkpoint not found: {e}. Creating a new run")

    wandb.login()
    wandb.init(
        project=args.wandb_project_name,
        name="test_emuru", 
        config=vars(args)
    )

    gen_test_image(
        eval_loader,
        model, 
        accelerator,
        weight_dtype,
        total_image=1000,
        text_path=f"{dataset_dir}/unique_words.txt",
        output_dir=args.output_dir
    )
    


if __name__ == "__main__":
    train()
