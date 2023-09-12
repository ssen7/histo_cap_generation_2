# Debugging Neural Networks: https://benjamin-computer.medium.com/debugging-neural-networks-6fa65742efd
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
from eval_utils import custom_evaluate, custom_evaluate_only_resnet
from nltk.translate.bleu_score import corpus_bleu

from pytorch_lightning import seed_everything
import argparse
import pdb

# seed_everything(42, workers=True) # done v14
# seed_everything(123, workers=True) # done v15
# seed_everything(43, workers=True) # done v16
# seed_everything(0, workers=True) # done v17
# seed_everything(1, workers=True) # done v18
# seed_everything(1234, workers=True) # done in v13


class LightningModel(L.LightningModule):
    def __init__(self, th_encoder, decoder, word_map, encoder_lr=1e-4, decoder_lr=4e-4, encoded_image_size=7):
        super().__init__()

        self.save_hyperparameters(ignore=["th_encoder","decoder","word_map"])

        self.encoder_lr = encoder_lr  # learning rate for encoder if fine-tuning
        self.decoder_lr = decoder_lr 
        self.th_encoder = th_encoder
        self.decoder = decoder
        self.word_map = word_map
        self.encoded_image_size = encoded_image_size
        
        self.val_hypotheses=[]
        self.val_references=[]
        
        self.test_hypotheses=[]
        self.test_references=[]
        
        self.val_bleu = torchmetrics.BLEUScore()
        self.test_bleu = torchmetrics.BLEUScore()

        self.test_step_outputs=dict()
    
    def loss(self, output, target, alphas):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()
        # check_nan_alpha=torch.sum(torch.isnan(alphas))
        # check_nan_output=torch.sum(torch.isnan(output))
        # if check_nan_alpha | check_nan_output:
            # print(check_nan_output, check_nan_output)
        return loss
    
    def training_step(self, batch, batch_idx):
        th_img, caps, caplens = batch
        th_enc_out = self.th_encoder(th_img)

        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(th_enc_out, th_enc_out, caps, caplens)
        targets = caps_sorted[:, 1:]
        
        scores, _,_,_ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _,_,_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = self.loss(scores, targets, alphas)
        
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        
        top5 = accuracy(scores, targets, 5)
        self.log("top5_train_acc", top5, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        th_img, caps, caplens, allcaps = batch
        th_enc_out = self.th_encoder(th_img)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(th_enc_out, th_enc_out, caps, caplens)
        targets = caps_sorted[:, 1:]
        
        scores_copy = scores.clone()
        scores, _,_,_ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _,_,_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        
        loss = self.loss(scores, targets, alphas)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        # References
        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            self.val_references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        self.val_hypotheses.extend(preds)
    
    def on_validation_epoch_end(self):

        references = self.val_references
        references = [item for sublist in references for item in sublist]
        hypotheses = self.val_hypotheses
               
        # print("Hypotheses: {}".format(hypotheses))
        # print("References: {}".format(references))
        hypotheses_str=return_string_list(hypotheses)
        references_str=[[x] for x in return_string_list(references)]
        # print("Hypotheses: {}".format(hypotheses_str))
        # print("References: {}".format(references_str))
        self.val_bleu(target=references_str, preds=hypotheses_str)
        # print(self.val_bleu(target=references_str, preds=hypotheses_str))
        self.log("val_bleu", self.val_bleu, prog_bar=True, on_epoch=True, on_step=False)
        
        # clean out created lists at end of validation epoch
        self.val_hypotheses=[]
        self.val_references=[]

    def test_step(self, batch, batch_idx):
        self.test_references, self.test_hypotheses = custom_evaluate_only_resnet(batch, self.th_encoder, 
                                                                     self.decoder, self.word_map, self.device, 
                                                                     self.test_references, self.test_hypotheses, k=1, encoded_image_size=self.encoded_image_size)

    
    def on_test_epoch_end(self):
        references, hypotheses = self.test_references, self.test_hypotheses
        references = [item for sublist in references for item in sublist]
        hypotheses_str=return_string_list(hypotheses)
        references_str=[[x] for x in return_string_list(references)]
        
        # save preds of test to dataframe
        df=pd.DataFrame()
        df['preds']=hypotheses_str
        df['target']=references_str
        df.to_csv(os.path.join(self.logger.log_dir, "saved_test.csv"), index=False)
        
        # print("Hypotheses: {}".format(hypotheses_str))
        # print("References: {}".format(references_str))
        self.test_bleu(target=references_str, preds=hypotheses_str)
        # print(self.test_bleu(target=references_str, preds=hypotheses_str))
        self.log("test_bleu", self.test_bleu, prog_bar=True)
        
        self.test_hypotheses=[]
        self.test_references=[]

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, self.decoder.parameters())},
                {'params': filter(lambda p: p.requires_grad, self.th_encoder.parameters()), 'lr': self.encoder_lr}
            ], lr=self.decoder_lr)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=8),
            "monitor": "val_bleu",
            "frequency": 1,
            "interval":"epoch"
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
            },
        }

if __name__=='__main__':
    print(watermark(packages="torch,pytorch_lightning,transformers", python=True), flush=True)
    print("Torch CUDA available?", torch.cuda.is_available(), flush=True)

    word_map=read_json('./data_files/word_map.json')
    rev_word_map = {v: k for k, v in word_map.items()}

    # df_path='/home/ss4yd/new_lstm_decoder/data_files/prepared_prelim_data_tokenized_cls256_pathcap_thumb.pickle'
    df_path='/home/ss4yd/new_lstm_decoder/data_files/prepared_prelim_data_tokenized_cls256_pathcap_thumb_finalv2_scr.pickle'

    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    encoder_dim = 512
    encoded_image_size=14 # >1 throws training run time error.. no idea why vanishing/exploding gradient prob

    batch_size=32
    epochs=120
    encoder_lr=1e-4
    decoder_lr=4e-4
    num_workers=4
    

    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       encoder_dim=encoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)

    thumb_enc=Encoder(encoded_image_size=encoded_image_size, fine_tune=True, img_net_pre_train=False, random_init=True)
    
    train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    
    val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    train_loader = torch.utils.data.DataLoader(ResnetDataset(df_path,'train', th_transform=train_transform), batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(ResnetDataset(df_path,'val', th_transform=val_transform), batch_size=batch_size, shuffle=False,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(ResnetDataset(df_path,'test', th_transform=val_transform), batch_size=1, shuffle=False,num_workers=num_workers)

    lightning_model = LightningModel(thumb_enc, decoder, word_map, encoder_lr=encoder_lr, decoder_lr=decoder_lr, encoded_image_size=encoded_image_size)
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="val_bleu", filename='{epoch}-{val_bleu:.2f}-{step:.2f}'),  # save top 1 model
        EarlyStopping(monitor="val_bleu", min_delta=0.000, patience=20, verbose=False, mode="max"),
        # StochasticWeightAveraging(swa_lrs=1e-2)
    ]

    logger = CSVLogger(save_dir="/scratch/ss4yd/logs_only_resnet_random_init/", name="my_model")

    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        precision='16-mixed',
        logger=logger,
        log_every_n_steps=100,
        deterministic=False,
        gradient_clip_val=5.0,
        gradient_clip_algorithm="value",
        # accumulate_grad_batches=32,
        # detect_anomaly=True,
        # limit_train_batches=0.2, 
        # limit_val_batches=0.02,
        # limit_test_batches=0.01
    )

    start = time.time()
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    test_bleu = trainer.test(lightning_model, test_loader, ckpt_path="best")

    with open(os.path.join(trainer.logger.log_dir, "outputs.txt"), "w") as f:
        f.write((f"Time elapsed {elapsed/60:.2f} min\n"))
        f.write(f"Test BLEU-4: {test_bleu}")

