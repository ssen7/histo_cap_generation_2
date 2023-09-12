import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torchmetrics
from watermark import watermark

import os
import time

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention, VITEncoder
from dataloader import *
from utils import *
from eval_utils import custom_evaluate, custom_evaluate_only_resnet, custom_evaluate_only_resnet_plus, custom_evaluate_only_vit
from training_script_only_vit import LightningModel

word_map=read_json('./data_files/word_map.json')
rev_word_map = {v: k for k, v in word_map.items()}

df_path='/home/ss4yd/new_lstm_decoder/data_files/prepared_prelim_data_tokenized_cls256_pathcap_thumb_finalv2_scr.pickle'

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
encoder_dim = 512
encoded_image_size=14 # >1 throws training run time error.. no idea why vanishing/exploding gradient prob
vit_img_size = 64

batch_size=32
epochs=120
encoder_lr=1e-4
decoder_lr=4e-4

num_workers=10

decoder = DecoderWithAttention(attention_dim=attention_dim,
                                    embed_dim=emb_dim,
                                    decoder_dim=decoder_dim,
                                    encoder_dim=encoder_dim,
                                    vocab_size=len(word_map),
                                    dropout=dropout)

thumb_enc=Encoder(encoded_image_size=encoded_image_size, fine_tune=True, img_net_pre_train=True)
vit_enc = VITEncoder(pretrain_4k='vit4k_xs_dino', freeze_4k=True)

val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

test_loader = torch.utils.data.DataLoader(ResnetPlusVitDataset(df_path,'test', th_transform=val_transform, vit_img_size=vit_img_size), 
                                            batch_size=1, shuffle=False, collate_fn=val_collate_fn, num_workers=num_workers)

lightning_model = LightningModel(vit_enc, decoder, word_map, encoder_lr=encoder_lr, decoder_lr=decoder_lr, encoded_image_size=encoded_image_size)

trainer = L.Trainer(limit_test_batches=1.0)

print(trainer.test(lightning_model, test_loader, ckpt_path="/scratch/ss4yd/logs_only_vit/my_model/version_4/checkpoints/epoch=13-val_bleu=0.22-step=3500.00.ckpt"))


# model = LightningModel.load_from_checkpoint("/scratch/ss4yd/logs_only_vit/my_model/version_4/checkpoints/epoch=13-val_bleu=0.22-step=3500.00.ckpt", *[vit_enc,decoder,word_map])
# print(model)