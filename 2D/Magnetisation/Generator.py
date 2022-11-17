"""
This modules contains the neural newtorks to reconstruct the source from the measured quantity 

"""

__author__ = "Adrien Dubois and David Broadway"

# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
class generator_CNN(nn.Module):
    # class to create a convolutionnal neural network for magnetisation reconstruction

    def __init__(self, Size=2, ImageSize=256, kernel = 5, stride = 2, padding = 2, channels_in = 1, channels_out=1):
        super(generator_CNN, self).__init__()

        M=Size

        if ImageSize == 512:
            ConvolutionSize = 32
        elif ImageSize == 256:
            ConvolutionSize = 16
        else: # size is 128
            ConvolutionSize = 8
        # first index is the number of channels
        self.convi = nn.Conv2d(channels_in, 8*M, kernel, 1, padding)
        self.conv_r0 = nn.Conv2d(1, 8*M, kernel, 1, padding)
        self.conv1 = nn.Conv2d(8*M, 8*M, kernel, stride, padding)
        self.bn1  = nn.BatchNorm2d(8*M)
        self.conv2 = nn.Conv2d(8*M, 16*M, kernel, stride, padding)
        self.bn2  = nn.BatchNorm2d(16*M)
        self.conv3 = nn.Conv2d(16*M, 32*M, kernel, stride, padding)
        self.bn3  = nn.BatchNorm2d(32*M)
        self.conv4 = nn.Conv2d(32*M, 64*M, kernel, stride, padding)
        self.bn4  = nn.BatchNorm2d(64*M)

        self.conv5 = nn.Conv2d(64*M, 128*M, 5, 1, 2)
        self.bn5  = nn.BatchNorm2d(128*M)

        self.trans1 = nn.ConvTranspose2d(128*M, 64*M, kernel, stride, padding,1)
        self.trans2 = nn.ConvTranspose2d(64*M+32*M, 32*M, kernel, stride, padding,1)
        self.trans3 = nn.ConvTranspose2d(32*M+16*M, 16*M, kernel, stride, padding,1)
        self.trans4 = nn.ConvTranspose2d(16*M+8*M, 8*M, kernel, stride, padding,1)
        self.conv6 = nn.Conv2d(8*M, channels_out, kernel, 1, padding)
        self.conv7 = nn.Conv2d(channels_out, channels_out, kernel, 1, padding)
        # self.conv6 = nn.Conv2d(8*M, 2, 5, 1, 2)
        # self.conv7 = nn.Conv2d(2, 1, kernel, 1, padding)
        # add the 2 at the first index for 2 outputs. 

        self.fc11 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc12 = nn.Linear(120, 84)
        self.fc13 = nn.Linear(84, 1)

        self.fc21 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc22 = nn.Linear(120, 84)
        self.fc23 = nn.Linear(84, 1)

        self.fc31 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc32 = nn.Linear(120, 84)
        self.fc33 = nn.Linear(84, 1)

        self.fc41 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc42 = nn.Linear(120, 84)
        self.fc43 = nn.Linear(84, 1)

        self.fc51 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc52 = nn.Linear(120, 84)
        self.fc53 = nn.Linear(84, 1)

        self.transfc1 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.transfc2 = nn.Linear(120, 256)
        self.transfc3 = nn.Linear(256, 65536)
        
        # Get the optimizer for the CNN instance
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self,input,
                roi,
                Nv_theta=False,
                Nv_phi=False,
                Nv_Height=False,
                M_angle=False,
                PositiveMagnetisationOnly = False,
                IntegerOnly=False, 
                MagnetisationLayerRange = None):

        conv0 = self.convi(input)
        roi0 = self.conv_r0(roi)
        mask = roi*conv0
        conv0 = F.leaky_relu(mask,0.2)
        conv1 = F.leaky_relu(self.bn1(self.conv1(conv0)),0.2)
        conv2 = F.leaky_relu(self.bn2(self.conv2(conv1)),0.2)
        conv3 = F.leaky_relu(self.bn3(self.conv3(conv2)),0.2)
        conv4 = F.leaky_relu(self.bn4(self.conv4(conv3)),0.2)

        conv5 = F.leaky_relu(self.conv5(conv4),0.2)

        trans1 = F.leaky_relu(self.bn4(self.trans1(conv5)),0.2)
        trans2 = F.leaky_relu(self.bn3(self.trans2(torch.cat([conv3,trans1], dim=1))),0.2)
        trans3 = F.leaky_relu(self.bn2(self.trans3(torch.cat([conv2,trans2], dim=1))),0.2)
        trans4 = F.leaky_relu(self.bn1(self.trans4(torch.cat([conv1,trans3], dim=1))),0.2)

        conv6 = self.conv6(trans4)
        conv7 = self.conv7(conv6)
        
        if PositiveMagnetisationOnly is True:
            conv8 = F.relu(conv7)
            #print('PositiveMagnetisationOnly')
        else:
            conv8 = (conv7)
        
        MagnetisationTheta = torch.flatten(conv4, 1)
        MagnetisationTheta= F.leaky_relu(self.fc11(MagnetisationTheta),0)
        MagnetisationTheta= F.leaky_relu(self.fc12(MagnetisationTheta),0.)
        MagnetisationTheta= (self.fc13(MagnetisationTheta))

        MagnetisationPhi = torch.flatten(conv4, 1)
        MagnetisationPhi = F.leaky_relu(self.fc21(MagnetisationPhi),0)
        MagnetisationPhi = F.leaky_relu(self.fc22(MagnetisationPhi),0.)
        MagnetisationPhi = (self.fc23(MagnetisationPhi))

        NVTheta = torch.flatten(conv4, 1)
        NVTheta= F.leaky_relu(self.fc31(NVTheta))
        NVTheta= F.leaky_relu(self.fc32(NVTheta))
        NVTheta= self.fc33(NVTheta).abs()

        NVPhi = torch.flatten(conv4, 1)
        NVPhi= F.leaky_relu(self.fc41(NVPhi))
        NVPhi= F.leaky_relu(self.fc42(NVPhi))
        NVPhi= self.fc43(NVPhi)

        MInt = torch.flatten(conv4, 1)
        MInt= F.leaky_relu(self.fc51(MInt))
        MInt= F.leaky_relu(self.fc52(MInt))
        MInt= self.fc53(MInt)
        MInt = torch.abs(MInt) 
        
        if IntegerOnly is True:
            if MagnetisationLayerRange is not None:
                MRange = MagnetisationLayerRange[1] - MagnetisationLayerRange[0]
                if MInt == 0:
                    MInt = MInt + MagnetisationLayerRange[0] + MRange/2
                if MInt < MagnetisationLayerRange[0]:
                    Mdiff = MagnetisationLayerRange[0] - MInt
                    MInt = MInt + Mdiff + MRange/2
                    
                if MInt > MagnetisationLayerRange[1]:
                    Mdiff = MInt - MagnetisationLayerRange[1] 
                    MInt = MInt - Mdiff - MRange/2
            
            conv9 = torch.round(roi*conv8/MInt)
            conv9 = conv9*MInt
            
        else:
            conv9 = roi*(conv8)

        return MInt,conv9,MagnetisationTheta,MagnetisationPhi,NVTheta,NVPhi

class generator_MLP(nn.Module):
    # class to create a fully connected neural network for magnetisation reconstruction
    def __init__(self):
        super(generator_MLP, self).__init__()

        self.enc1 = nn.Linear(in_features=65536, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)
        self.enc6 = nn.Linear(in_features=16, out_features=1)

        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=65536)

        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self,input,
                roi,
                Nv_theta=False,
                Nv_phi=False,
                Nv_Height=False,
                M_angle=False,
                PositiveMagnetisationOnly = False,
                IntegerOnly=False, 
                MagnetisationLayerRange = None):

        enc1 = F.relu(self.enc1(input))
        enc2 = F.relu(self.enc2(enc1))
        enc3 = F.relu(self.enc3(enc2))
        enc4 = F.relu(self.enc4(enc3))
        enc5 = F.relu(self.enc5(enc4))
        enc6 = F.relu(self.enc6(enc5))

        dec1 = F.relu(self.dec1(enc5))
        dec2 = F.relu(self.dec2(dec1))
        dec3 = F.relu(self.dec3(dec2))
        dec4 = F.relu(self.dec4(dec3))
        dec5 = (self.dec5(dec4))

        if IntegerOnly is True:
            dec6 = dec5*roi
        else:
            dec6 = dec5*roi

        return enc6,dec6,dec5,dec5,dec5,dec5


class generator_CNN_J(nn.Module):
    # class to create a convolutionnal neural  neural network for current density reconstruction

    def __init__(self, Size=1,ImageSize=128, kernel=5,stride=2,padding=2):
        super(generator_CNN_J, self).__init__()

        self.convi = nn.Conv2d(1, 8*Size, kernel, 1, padding)
        self.conv_r0 = nn.Conv2d(1, 8*Size, kernel, 1, padding)
        self.conv1 = nn.Conv2d(8*Size, 8*Size, kernel, stride, padding)
        self.bn1  = nn.BatchNorm2d(8*Size)
        self.conv2 = nn.Conv2d(8*Size, 16*Size, kernel, stride, padding)
        self.bn2  = nn.BatchNorm2d(16*Size)
        self.conv3 = nn.Conv2d(16*Size, 32*Size, kernel, stride, padding)
        self.bn3  = nn.BatchNorm2d(32*Size)
        self.conv4 = nn.Conv2d(32*Size, 64*Size, kernel, stride, padding)
        self.bn4  = nn.BatchNorm2d(64*Size)

        self.conv5 = nn.Conv2d(64*Size, 128*Size, 5, 1, 2)
        self.bn5  = nn.BatchNorm2d(128*Size)

        self.trans1 = nn.ConvTranspose2d(128*Size, 64*Size, kernel, stride, padding,1)
        self.trans2 = nn.ConvTranspose2d(64*Size+32*Size, 32*Size, kernel, stride, padding,1)
        self.trans3 = nn.ConvTranspose2d(32*Size+16*Size, 16*Size, kernel, stride, padding,1)
        self.trans4 = nn.ConvTranspose2d(16*Size+8*Size, 8*Size, kernel, stride, padding,1)
        self.conv6 = nn.Conv2d(8*Size, 1, 5, 1, 2)
        self.conv7 = nn.Conv2d(1, 1, kernel, 1, padding)

        # Get the optimizer for the CNN instance
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self,input,roi):

        conv0 = self.convi(input)
        roi0 = self.conv_r0(roi)
        mask = roi0*conv0
        conv0 = F.leaky_relu(mask,0.2)
        conv1 = F.leaky_relu(self.bn1(self.conv1(conv0)),0.2)
        conv2 = F.leaky_relu(self.bn2(self.conv2(conv1)),0.2)
        conv3 = F.leaky_relu(self.bn3(self.conv3(conv2)),0.2)
        conv4 = F.leaky_relu(self.bn4(self.conv4(conv3)),0.2)

        conv5 = F.leaky_relu(self.conv5(conv4),0.2)

        trans1 = F.leaky_relu(self.bn4(self.trans1(conv5)),0.2)
        trans2 = F.leaky_relu(self.bn3(self.trans2(torch.cat([conv3,trans1], dim=1))),0.2)
        trans3 = F.leaky_relu(self.bn2(self.trans3(torch.cat([conv2,trans2], dim=1))),0.2)
        trans4 = F.leaky_relu(self.bn1(self.trans4(torch.cat([conv1,trans3], dim=1))),0.2)

        conv6 = self.conv6(trans4)
        conv7 = F.relu(self.conv7(conv6)*roi)

        conv0_2 = self.convi(input)
        roi0_2 = self.conv_r0(roi)
        mask_2 = roi0_2*conv0_2
        conv0_2 = F.leaky_relu(mask_2,0.2)
        conv1_2 = F.leaky_relu(self.bn1(self.conv1(conv0_2)),0.2)
        conv2_2 = F.leaky_relu(self.bn2(self.conv2(conv1_2)),0.2)
        conv3_2 = F.leaky_relu(self.bn3(self.conv3(conv2_2)),0.2)
        conv4_2 = F.leaky_relu(self.bn4(self.conv4(conv3_2)),0.2)
        conv5_2 = F.leaky_relu(self.conv5(conv4_2),0.2)

        trans1_2 = F.leaky_relu(self.bn4(self.trans1(conv5_2)),0.2)
        trans2_2 = F.leaky_relu(self.bn3(self.trans2(torch.cat([conv3_2,trans1_2], dim=1))),0.2)
        trans3_2 = F.leaky_relu(self.bn2(self.trans3(torch.cat([conv2_2,trans2_2], dim=1))),0.2)
        trans4_2 = F.leaky_relu(self.bn1(self.trans4(torch.cat([conv1_2,trans3_2], dim=1))),0.2)

        conv6_2 = self.conv6(trans4_2)
        conv7_2 = F.relu(self.conv7(conv6_2)*roi)



        return conv7,conv7_2