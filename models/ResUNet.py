import torch
import torch.nn as nn

# Residual Convolutional Block used in the network


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        Initializes the residual convolution block with two convolutional layers,
        each followed by a ReLU activation. If the input and output channels don't match,
        a 1x1 convolution is used to adjust the input channels to the output channels.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.adjust_channels = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1) if in_ch != out_ch else None

    def forward(self, x):
        """
        Forward pass through the residual convolution block.
        The input is passed through two convolutions, and the residual connection
        is added to the output.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying convolutions and adding the residual.
        """
        residual = x if self.adjust_channels is None else self.adjust_channels(
            x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + residual)

# ResUNet architecture definition


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        Initializes the ResUNet model with an encoder-decoder architecture,
        using residual convolution blocks in both the encoder and decoder.

        Args:
            in_channels (int): Number of input channels (default is 3 for RGB images).
            out_channels (int): Number of output channels (default is 1 for binary segmentation).
        """
        super(ResUNet, self).__init__()
        self.enc1 = ResConvBlock(in_channels, 64)
        self.enc2 = ResConvBlock(64, 128)
        self.enc3 = ResConvBlock(128, 256)
        self.enc4 = ResConvBlock(256, 512)
        self.enc5 = ResConvBlock(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResConvBlock(512 + 512, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResConvBlock(256 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResConvBlock(128 + 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResConvBlock(64 + 64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the entire ResUNet architecture.
        The input goes through the encoder, then the decoder, with skip connections
        from the encoder to the corresponding decoder layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after the final convolution, passed through a sigmoid for binary classification.
        """
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        dec4 = self.dec4(torch.cat([self.up4(enc5), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return torch.sigmoid(self.final(dec1))
