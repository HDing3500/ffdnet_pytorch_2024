import torch.nn as nn
from torch.autograd import Variable
import functions

class UpSampleFeatures(nn.Module):
    """Implements the last layer of FFDNet"""
    def __init__(self):
        super(UpSampleFeatures, self).__init__()

    def forward(self, x):
        return functions.upsamplefeatures(x)


class IntermediateDnCNN(nn.Module):
    """Implements the middle part of the FFDNet architecture, which is basically a DnCNN net"""
    def __init__(self, input_features, middle_features, num_conv_layers):
        super(IntermediateDnCNN, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features

        # Output features based on input size
        if self.input_features == 5:
            self.output_features = 4  # Grayscale image
        elif self.input_features == 15:
            self.output_features = 12  # RGB image
        elif self.input_features == 4:
            self.output_features = 12  # RGB + noise map
        else:
            raise Exception('Invalid number of input features')

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_features,
                                out_channels=self.middle_features,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.num_conv_layers-2):
            layers.append(nn.Conv2d(in_channels=self.middle_features,
                                    out_channels=self.middle_features,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.middle_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.middle_features,
                                out_channels=self.output_features,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))
        self.itermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.itermediate_dncnn(x)
        return out


class FFDNet(nn.Module):
    def __init__(self, num_input_channels):
        super(FFDNet, self).__init__()

        self.num_input_channels = num_input_channels
        if self.num_input_channels == 1:
            # Grayscale image
            self.num_feature_maps = 64
            self.num_conv_layers = 15
            self.downsampled_channels = 5
            self.output_features = 4
        elif self.num_input_channels == 3:
            # RGB image
            self.num_feature_maps = 96
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.output_features = 12
        elif self.num_input_channels == 4:
            # RGB + noise map (4 channels)
            self.num_feature_maps = 96
            self.num_conv_layers = 12
            self.downsampled_channels = 4  # Use 4 input channels
            self.output_features = 12
        else:
            raise Exception('Invalid number of input features')

        # Initialize the intermediate DnCNN part
        self.intermediate_dncnn = IntermediateDnCNN(input_features=self.downsampled_channels,
                                                    middle_features=self.num_feature_maps,
                                                    num_conv_layers=self.num_conv_layers)
        
        # Initialize upsampling layer
        self.upsamplefeatures = UpSampleFeatures()

    def forward(self, x):
        # Assuming x is a tensor with 4 channels (RGB + noise map)
        concat_noise_x = x
        
        # Pass through intermediate layers
        h_dncnn = self.intermediate_dncnn(concat_noise_x)
        
        # Upsample the output
        pred_noise = self.upsamplefeatures(h_dncnn)
        
        return pred_noise
