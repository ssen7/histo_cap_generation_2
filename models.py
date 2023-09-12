import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import torch.nn.functional as F
import pdb
from os.path import join
from collections import OrderedDict

from model_utils import *
from vision_transformer4k import vit4k_xs

VIT_CKPT_PATH = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints'
RESNET_CKPT_PATH = '/home/ss4yd/new_lstm_decoder/self_supervised_ckpts/tenpercent_resnet18.ckpt'

# ref: https://github.com/zhangrenyuuchicago/PathCap/blob/master/att_thumbnail_tiles/models.py
class Encoder(nn.Module):
    """
    Thumbnail encoder.
    """
    def __init__(self, encoded_image_size=14, img_net_pre_train=False, random_init=False, fine_tune=False):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        if img_net_pre_train:
            resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            modules = list(resnet.children())[:-2]
            self.resnet = nn.Sequential(*modules)
        elif random_init:
            resnet = torchvision.models.resnet18()
            modules = list(resnet.children())[:-2]
            self.resnet = nn.Sequential(*modules)
        else:
            resnet = torchvision.models.__dict__['resnet18'](weights=None)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.load(RESNET_CKPT_PATH, map_location=device)
            state_dict = state['state_dict']
            for key in list(state_dict.keys()):
                state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
            resnet = load_model_weights(resnet, state_dict)
            modules = list(resnet.children())[:-2]
            self.resnet = nn.Sequential(*modules)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune(fine_tune)

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)  
        out = out.permute(0, 2, 3, 1)  
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

# ref: https://github.com/mahmoodlab/HIPT/tree/b5f4844f2d8b013d06807375166817eeb939a5aa/HIPT_4K
### V3 Encoder - Increased BLEU-4 ###
class VITEncoder(nn.Module):
    def __init__(self, path_input_dim=384,  size_arg = "small", dropout=0.25, pretrain_4k='None', freeze_4k=False, pretrain_WSI='None', freeze_WSI=False):
        super(VITEncoder, self).__init__()
        self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
        #self.fusion = fusion
        size = self.size_dict_path[size_arg]

        ### Local Aggregation
        self.local_vit = vit4k_xs()
        if pretrain_4k != 'None':
            print("Loading Pretrained Local VIT model...",)
            state_dict = torch.load(f'{VIT_CKPT_PATH}/%s.pth' % pretrain_4k, map_location='cpu')['teacher']
            state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
            state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.local_vit.load_state_dict(state_dict, strict=False)
            print("Done!")
        if freeze_4k:
            print("Freezing Pretrained Local VIT model")
            for param in self.local_vit.parameters():
                param.requires_grad = False
            print("Done")

        ### Global Aggregation
        self.pretrain_WSI = pretrain_WSI
        if pretrain_WSI != 'None':
            pass
        else:
            self.global_phi = nn.Sequential(nn.Linear(576, 576), nn.ReLU(), nn.Dropout(dropout))
            # self.global_transformer = nn.TransformerEncoder(
            #     nn.TransformerEncoderLayer(
            #         d_model=576, nhead=3, dim_feedforward=192, dropout=dropout, activation='relu'
            #     ), 
            #     num_layers=2
            # )
            self.global_attn_pool = Attn_Net_Gated(L=576, D=192, dropout=dropout, n_classes=1)
            self.global_rho = nn.Sequential(*[nn.Linear(576, 512), nn.ReLU(), nn.Dropout(dropout)])
        

    def forward(self, x_256, **kwargs):
        ### Local
        h_4096 = self.local_vit(x_256.unfold(1, 16, 16).transpose(1,2))
        features_mean256 = x_256.mean(dim=1)
        h_4096 = torch.cat([features_mean256, h_4096], dim=1)

        ### Global
        if self.pretrain_WSI != 'None':
            h_WSI = self.global_vit(h_4096.unsqueeze(dim=0))
        else:
            h_4096 = self.global_phi(h_4096)
            # print(h_4096.shape)
            # h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
            # print(h_4096.shape)
            A_4096, h_4096 = self.global_attn_pool(h_4096)  
            A_4096 = torch.transpose(A_4096, 1, 0)
            A_4096 = F.softmax(A_4096, dim=1) 
            h_path = torch.mm(A_4096, h_4096)
            h_WSI = self.global_rho(h_path)

        return h_WSI, A_4096

        # .unsqueeze(0).unsqueeze(0).repeat(1,self.encoded_image_size,self.encoded_image_size,1)

# ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, th_enc_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        th_enc_out = th_enc_out.view(batch_size, -1, encoder_dim) # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        th_enc_out = th_enc_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(th_enc_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind