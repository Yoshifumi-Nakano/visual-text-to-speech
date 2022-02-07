import torch
import torch.nn as nn
import numpy as np

class VisualFeatureExtractor(nn.Module):
    def __init__(self,
                 slice_width,
                 slice_height,
                 embed_dim,
                 stride,
                 embed_normalize=False,
                 bridge_relu=False,
                 kernel_size=(3,3),
                 num_convolutions=1):

        super().__init__()
        self.slice_width = slice_width
        self.slice_height = slice_height
        self.embed_dim = embed_dim
        self.embed_normalize = embed_normalize
        self.bridge_relu = bridge_relu
        self.kernel_size = (slice_height,kernel_size[1]) if kernel_size[0]==-1 else kernel_size
        self.num_convolutions = num_convolutions
        self.stride=stride

        # we require odd kernel size values:
        assert self.kernel_size[0] % 2 != 0, f"conv2d kernel height {self.kernel_size} is even. we require odd. did you use {self.slice_height}?"
        assert self.kernel_size[1] % 2 != 0, f"conv2d kernel width {self.kernel_size} is even. we require odd."

        # padding for dynamic kernel size. assumes we do not change the conv dilation or conv stride (we currently use the defaults)
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

        #slice tensor image
        image_slice=[]
        for image in images:
            tensors = []
            for i in range(0,width-1,self.stride):
                slice_tensor = image[:,:,i:i+self.stride]
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