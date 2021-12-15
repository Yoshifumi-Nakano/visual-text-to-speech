import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols

class NLayerImageCNN(nn.Module):
    def __init__(self,
                 slice_width,
                 slice_height,
                 embed_dim,
                 stride,
                 width,
                 height,
                 embed_normalize=False,
                 bridge_relu=False,
                 kernel_size=(3,3),
                 num_convolutions=1):

        super().__init__()
        self.embed_dim = embed_dim
        self.embed_normalize = embed_normalize
        self.bridge_relu = bridge_relu
        self.kernel_size = kernel_size
        self.num_convolutions = num_convolutions

        #画像の情報
        self.slice_width = slice_width
        self.slice_height = slice_height
        self.stride=stride
        self.width=width
        self.height=height

        # kernel sizeは奇数
        assert self.kernel_size[0] % 2 != 0, f"conv2d kernel height {self.kernel_size} is even. we require odd. did you use {self.slice_height}?"
        assert self.kernel_size[1] % 2 != 0, f"conv2d kernel width {self.kernel_size} is even. we require odd."

        # kernelが3ならpaddingは1
        padding_h = int((kernel_size[0] - 1) / 2)
        padding_w = int((kernel_size[1] - 1) / 2)
       
        ops = []
        for i in range(num_convolutions):
            ops.append(nn.Conv2d(1, 1, stride=1, kernel_size=self.kernel_size, padding=(padding_h,padding_w)))
            if embed_normalize:
                ops.append(nn.BatchNorm2d(1)),
            ops.append(nn.ReLU(inplace=True))

        self.embedder = nn.Sequential(*ops)

        if self.bridge_relu:
            self.bridge = nn.Sequential(
                nn.Linear(slice_width * slice_height, embed_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.bridge = nn.Linear(slice_width * slice_height, embed_dim)

        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

    def forward(self, images):
        batch_size, channels, height, width = images.shape

        #tensor画像をsliceする
        image_slice=[]
        for image in images:
            tensors = []
            #白色でpadding
            image = F.pad(image, ((self.slice_width-self.width)//2, (self.slice_width-self.width)//2), "constant", 1) 

            for i in range(0,width-1,self.stride):
                slice_tensor = image[:,:,i:i+self.slice_width]
                tensors.append(slice_tensor)
            image_slice.append(torch.stack(tensors))
        image_slice=torch.stack(image_slice)
        batch_size, src_len, channels, height, width = image_slice.shape
        

        pixels = image_slice.view(batch_size * src_len, channels, height, width)

        # Embed and recast to 3d tensor
        embeddings = self.embedder(pixels)
        embeddings = embeddings.view(batch_size * src_len, height * width)
        embeddings = self.bridge(embeddings)
        embeddings = embeddings.view(batch_size, src_len, self.embed_dim)

        return embeddings

        
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        #image config
        width=config["image"]["width"]
        height=config["image"]["height"]
        stride=config["image"]["stride"]
        slice_height=config["image"]["slice_height"]
        slice_width=config["image"]["slice_width"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )

        self.src_accent_emb = nn.Embedding(
            4, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        #width ひらがな一枚あたりの画像の横幅
        #height ひらがな一枚あたりの画像の縦幅
        #slice_width 横幅何枚ごとにsliceしていくか
        #slice_height 縦幅何枚ごとにsliceしていくか
        self.NLayerImgageCNN=NLayerImageCNN(slice_width=slice_width,slice_height=slice_height,embed_dim=d_model,embed_normalize=True,bridge_relu=True,kernel_size=(3,3),num_convolutions=1,stride=stride,width=width,height=height)

    def forward(self, src_seq, mask,accents=None, return_attns=False,images=None):
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]


        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            assert False
            if accents is not None:
                enc_output = self.src_word_emb(src_seq) + self.src_accent_emb(accents) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
                )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                    src_seq.device
                )
            else:
                enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
                )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                    src_seq.device
                )

        else:
            if accents is not None:
                assert False
                enc_output = self.src_word_emb(src_seq) + self.src_accent_emb(accents) +self.position_enc[
                    :, :max_len, :
                ].expand(batch_size, -1, -1)
            else:
                if images is None:
                    assert False
                    enc_output = self.src_word_emb(src_seq) +self.position_enc[
                        :, :max_len, :
                    ].expand(batch_size, -1, -1)
                else:
                    enc_output = self.NLayerImgageCNN(images) +self.position_enc[
                        :, :max_len, :
                    ].expand(batch_size, -1, -1)


        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask