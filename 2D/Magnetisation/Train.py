# -*- coding: utf-8 -*-

"""
Module to train the network
"""

__author__ = "Adrien Dubois and David Broadway"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from tqdm import tqdm
from Magnetisation.Evaluate import evaluate

class Magnetisation_CNN_training():
    
    def __init__(self, device, Generator, PROP,ML_options):
        self.device = device
        self.generator = Generator
        self.PROP = PROP
        self.ML_options=ML_options
        
        self.mask = np.where(PROP.MagneticFieldExtended == 0,0,1)  

        if ML_options['mlp'] is True:
            self.img_input=torch.FloatTensor(np.reshape(PROP.MagneticFieldExtended,(PROP.MagneticFieldExtended.shape[0]*PROP.MagneticFieldExtended.shape[0]))[np.newaxis])
            self.mask_t = torch.FloatTensor(np.reshape(self.mask,self.mask.shape[0]*self.mask.shape[0])[np.newaxis])
        else:
            self.img_input = torch.FloatTensor(PROP.MagneticFieldExtended[np.newaxis, np.newaxis, :, :])
            self.mask_t = torch.FloatTensor(self.mask[np.newaxis,np.newaxis])

        train_data_cnn = TensorDataset(self.img_input, self.mask_t)
        self.train_loader = DataLoader( dataset=(train_data_cnn))
    
    def train_cnn(self,   
                  mlp=False, 
                  LossFunction = 'L1',
                  Epochs = 100,
                  PositiveMagnetisationOnly = False,
                  IntegerOnly = False,
                  IntergerInitialTraining = 100,
                  Magnetization = None,
                  PrintLossValue = False,
                  MagnetisationLayerRange = None
                 ):   
        # training for the magnetization reconstruction

        ImageSize = self.PROP.options['ImageShape']
        L1_Loss = nn.L1Loss()
        L2_Loss = nn.MSELoss()

        if LossFunction == "L1":
            Loss = L1_Loss
        elif LossFunction == "L2":
            Loss = L2_Loss

        unit_conversion = 1e-18 / 9.27e-24

        running_loss = 0.0
        loss_values = []
        G_loss_List = []
        snr_List = []
        Errors = dict()

        for batch_idx, (data,mask_t) in enumerate(self.train_loader):
            data,mask_t= (data).to(self.device), mask_t.to(self.device)
            data=data*1

            for Epoch in tqdm(range(Epochs)):

                self.generator.train()

                if mlp:
                    data = torch.reshape(data,(1,self.PROP.MagneticFieldExtended.shape[0]*self.PROP.MagneticFieldExtended.shape[0]))
                    mask_t=torch.FloatTensor(self.PROP.Mask.reshape(1,self.mask.shape[0]*self.mask.shape[0]))

                if Epoch > IntergerInitialTraining and (Epoch % 2) == 0 and (IntegerOnly):
                    M,img,MagnetisationTheta,MagnetisationPhi,NVtheta,NVphi = self.generator(data,mask_t,PositiveMagnetisationOnly=PositiveMagnetisationOnly,IntegerOnly=True, MagnetisationLayerRange=MagnetisationLayerRange)
                else:
                    M,img,MagnetisationTheta,MagnetisationPhi,NVtheta,NVphi = self.generator(data,mask_t,PositiveMagnetisationOnly=PositiveMagnetisationOnly,IntegerOnly=False, MagnetisationLayerRange=MagnetisationLayerRange)

                reconstructed_M = img

                if  self.PROP.options['Magnetisation']['FindTheta']:
                    magnetisation_theta = MagnetisationTheta + 90
                else:
                    if 'Theta' in self.PROP.options['Magnetisation']:
                        magnetisation_theta = torch.from_numpy(np.array(self.PROP.options['Magnetisation']['Theta']))
                    else:
                        # If no magnetisation theta is given assume the magnetisation is out of plane
                        magnetisation_theta = torch.from_numpy(np.array(0.0))

                if  self.PROP.options['Magnetisation']['FindPhi']:
                    magnetisation_phi = MagnetisationPhi

                else:
                    if 'Phi' in self.PROP.options['Magnetisation']:
                        magnetisation_phi = torch.from_numpy(np.array(self.PROP.options['Magnetisation']['Phi']))
                    else:
                        magnetisation_phi = torch.from_numpy(np.array(90.0))

                self.PROP.define_magnetisation_transformation()
                d_matrix = torch.from_numpy(self.PROP.d_matrix).to(self.device)

                phi = torch.deg2rad(magnetisation_phi)
                theta = torch.deg2rad(magnetisation_theta)

                mx_mag = torch.cos(phi) * torch.sin(theta)
                my_mag = torch.sin(phi) * torch.sin(theta)
                mz_mag = torch.cos(theta)

                # Define the transoformation for bx, by, and bz
                m_to_bx = mx_mag * d_matrix[0, 0, ::] + my_mag * d_matrix[1, 0, ::] + mz_mag * d_matrix[2, 0, ::]
                m_to_by = mx_mag * d_matrix[0, 1, ::] + my_mag * d_matrix[1, 1, ::] + mz_mag * d_matrix[2, 1, ::]
                m_to_bz = mx_mag * d_matrix[0,2, ::] + my_mag * d_matrix[1, 2, ::] + mz_mag * d_matrix[2, 2, ::]

                # Remove any nans or ints in the transformation
                tmp_m_to_bx= torch.logical_or(torch.isnan(m_to_bx), torch.isinf(m_to_bx)).to(self.device)
                tmp_m_to_by= torch.logical_or(torch.isnan(m_to_by), torch.isinf(m_to_by)).to(self.device)
                tmp_m_to_bz= torch.logical_or(torch.isnan(m_to_bz), torch.isinf(m_to_bz)).to(self.device)

                m_to_bx[tmp_m_to_bx] = 0
                m_to_by[tmp_m_to_by] = 0
                m_to_bz[tmp_m_to_bz] = 0

                if mlp:
                    reconstructed_M = torch.reshape(reconstructed_M,(1,1,self.PROP.MagneticFieldExtended.shape[0],self.PROP.MagneticFieldExtended.shape[0]))
                    data = torch.reshape(data,(self.PROP.MagneticFieldExtended.shape[0],self.PROP.MagneticFieldExtended.shape[0]))

                #reconstructed_image = torch.zeros(data.shape).to(device)
                fft_mag_image = torch.fft.fft2(reconstructed_M).to(self.device)

                bx = torch.fft.ifft2(m_to_bx * fft_mag_image).real.to(self.device)
                by = torch.fft.ifft2(m_to_by * fft_mag_image).real.to(self.device)
                bz = torch.fft.ifft2(m_to_bz * fft_mag_image).real.to(self.device)


                # Get the NV angle or try to fit the NV angles
                if self.PROP.options['NV']['FindTheta']:
                    NVTheta = torch.deg2rad(NVtheta)
                else:
                    NVtheta = torch.from_numpy(np.array(self.PROP.options['NV']['Theta']))
                    NVTheta = torch.deg2rad(NVtheta)

                if self.PROP.options['NV']['FindPhi']:
                    NVPhi = torch.deg2rad(NVphi)
                else:
                    NVphi = torch.from_numpy(np.array(self.PROP.options['NV']['Phi']))
                    NVPhi = torch.deg2rad(NVphi)

                # Define the NV Orientation
                u_prop=  [torch.sin(NVTheta) * torch.cos(NVPhi), torch.sin(NVTheta) * torch.sin(NVPhi), torch.cos(NVTheta)]

                if  self.PROP.options['Magnetisation']['FindTheta'] or self.PROP.options['Magnetisation']['FindPhi']:
                    bnv =  u_prop[2]*bz
                else:
                    bnv =  u_prop[0]*bx +  u_prop[1]*by + u_prop[2]*bz

                if LossFunction == "L1":
                    G_loss= Loss(data, bnv)
                elif LossFunction == "L2":
                    G_loss =torch.mean((data - bnv)**2)

                SNR = (torch.max(reconstructed_M)-torch.min(reconstructed_M))/(torch.absolute(torch.std(reconstructed_M)))

                # Extract the error function into a list for plotting and returning from the training.
                # Both the L1 and L2 error functions are converted into units of Tesla
                if LossFunction == "L1":
                    G_loss_List.append(G_loss.item()*data.size(0)/unit_conversion)
                    snr_List.append(SNR.item()*data.size(0)/unit_conversion)
                elif LossFunction == "L2":
                    # Take the sqrt to make the output in units of Tesla
                    G_loss_List.append( np.sqrt(G_loss.item()*data.size(0))/ unit_conversion)
                    snr_List.append(SNR.item()*data.size(0)/unit_conversion)

                Errors['Loss Function'] = G_loss_List
                Errors['SNR'] = snr_List

                if Magnetization is None:
                    if PrintLossValue:
                        print('loss = ' +  str(G_loss.item()*data.size(0)))

                # If a magnetization image is given calculate a loss function between the NN and the given image.
                # This is just for comparison but in principle could be used as the NN loss function
                if Magnetization is not None:
                    M_loss = Loss(Magnetization, reconstructed_M)
                    if PrintLossValue:
                        print('loss = ' +  str(M_loss.item()*data.size(0)))

                    running_loss =+ M_loss.item()*data.size(0)
                    loss_values.append(running_loss)
                    Errors['Magnetisation Loss'] = loss_values

                self.generator.optimizer.zero_grad()
                G_loss.backward()
                #reconstructed.backward(grads['reconstructed_M'])
                self.generator.optimizer.step()

        # Plot the evolution of the loss function at the end of the train
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(12)
        plt.subplot(1,2,1)
        plt.plot(G_loss_List, label='Loss function')
        #plt.plot(snr_List, label='Loss function')
        plt.ylabel('Average difference, $\Delta B (T)$')
        plt.title('Error function evolution')
        plt.xlabel('Epochs')
        ax.legend()
        plt.legend
        plt.subplot(1,2,2)
        plt.plot(snr_List, label='Loss function')
        plt.ylabel('Standard divation, $\mu (T)$')
        plt.title('Error function evolution')
        plt.xlabel('Epochs')
        ax.legend()
        plt.legend


        NVTheta = torch.rad2deg(NVTheta)
        NVPhi = torch.rad2deg(NVPhi)
        # FOR AFTER UPGRADING TO A CLASS

        # Detatch the results and return them as properties of the class.
        self.MagTheta = magnetisation_theta.detach().cpu().numpy()
        self.MagPhi = magnetisation_phi.detach().cpu().numpy()
        self.MMag = M.detach().cpu().numpy()
        self.NVtheta = NVTheta.detach().cpu().numpy()
        self.NVphi = NVPhi.detach().cpu().numpy()
        self.TrainingErrors = Errors
    
        self.ReconMag = reconstructed_M[0,0,:,:].detach().cpu().numpy()
        self.ReconBnv = bnv[0,0,:,:].detach().cpu().numpy()
        
        return magnetisation_theta, magnetisation_phi, bnv, Errors, NVtheta, NVphi

    
    def extract_results(self, plotResults = True):

        '''
        This function is used to return the results of the training done in train_cnn
        inputs:
            device: The cpu to gpu device used for the training
            img_input:
            mask_t:
            G_A: The generator for the neural network
            PROP: The class instance for propagation
            b_cnn: The return values from the training function train_cnn
            plotResults (default = True_: Whether or not to plot the results from the machine learning

        '''

        # Unit converse is to take the B map from the NN and convert it back into units of Tesla
        unit_conversion_M_to_B =9.27e-24 / 1e-18

        Results = dict()
        MagnetisationMap = self.ReconMag
        ReconstructedBnv = self.ReconBnv

        Results["Errors"] = self.TrainingErrors
        Results["M theta"] = self.MagTheta
        Results["M phi"]= self.MagPhi
        Results["M Mag"] = self.MMag
        Results["NV theta"]= self.NVtheta
        Results["NV phi"]= self.NVphi


        # When applying the OriginalROI the data gets rotated. The np.rot90 function is there to fix this issue.
        Results["Original B"] = np.rot90(self.PROP.MagneticFieldExtended[self.PROP.OriginalROI] * unit_conversion_M_to_B, k=1)
        Results["Reconstructed B"] = np.rot90(ReconstructedBnv[self.PROP.OriginalROI] * unit_conversion_M_to_B , k=1)
        Results["Magnetisation"] = np.rot90(MagnetisationMap[self.PROP.OriginalROI] , k=1)


        if plotResults:
            fig = plt.figure()
            fig.set_figheight(6)
            fig.set_figwidth(24)

            ax1 = plt.subplot(1,4,1)

            Range = np.max(np.abs(1e6*Results["Original B"]))
            plt.imshow(1e6*Results["Original B"], cmap="seismic", vmin = -Range, vmax = Range)
            plt.colorbar(fraction=0.046, pad=0.04,label="B  ($\mu T)$")
            scalebar = ScaleBar(self.PROP.options['PixelSize'], location='lower left')
            ax1.add_artist(scalebar)
            plt.xticks([])
            plt.yticks([])
            plt.title('Magnetic Field')

            ax2 = plt.subplot(1,4,2)
            Range = np.max(np.abs(1e6*Results["Reconstructed B"]))
            plt.imshow(1e6*Results["Reconstructed B"], cmap="seismic", vmin = -Range, vmax = Range)
            scalebar = ScaleBar(self.PROP.options['PixelSize'], location='lower left')
            ax2.add_artist(scalebar)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(fraction=0.046, pad=0.04,label="B ($\mu T)$")
            plt.title('reconstructed magneticfield')

            ax3 = plt.subplot(1,4,3)
            PlotData = 1e6*(Results["Original B"] - Results["Reconstructed B"])
            Range = np.max(np.abs(PlotData))
            plt.imshow(PlotData, cmap="seismic", vmin = -Range, vmax = Range)
            scalebar = ScaleBar(self.PROP.options['PixelSize'], location='lower left')
            ax3.add_artist(scalebar)
            plt.xticks([])
            plt.yticks([])
            # plt.clim([-150,150])
            plt.colorbar(fraction=0.046, pad=0.04,label="B  ($\mu T)$")
            plt.title('reconstruction difference B')

            ax4 = plt.subplot(1,4,4)
            PlotData = Results["Magnetisation"] 
            Range = np.max(np.abs(PlotData))
            #plt.imshow(PlotData, cmap="PuOr", vmin = -Range, vmax = Range)
            plt.imshow(PlotData)
            scalebar = ScaleBar(self.PROP.options['PixelSize'], location='lower left')
            ax4.add_artist(scalebar)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(fraction=0.046, pad=0.04,label="M  ($\mu_B nm^2$)")
            plt.title('Reconstructed magnetisation')

        return Results

    ##########################################################################################
    #     For multiple channels
    ##########################################################################################
    def train_cnn_multiple_channels(self,   
                  mlp=False, 
                  LossFunction = 'L1',
                  Epochs = 100,
                  PositiveMagnetisationOnly = False,
                  IntegerOnly = False,
                  IntergerInitialTraining = 100,
                  Magnetization = None,
                  PrintLossValue = False,
                  MagnetisationLayerRange = None
                 ):   
        # training for the magnetization reconstruction

        ImageSize = self.PROP.options['ImageShape']
        L1_Loss = nn.L1Loss()
        L2_Loss = nn.MSELoss()

        if LossFunction == "L1":
            Loss = L1_Loss
        elif LossFunction == "L2":
            Loss = L2_Loss

        unit_conversion = 1e-18 / 9.27e-24

        running_loss = 0.0
        loss_values = []
        G_loss_List = []
        snr_List = []
        Errors = dict()

        for batch_idx, (data,mask_t) in enumerate(self.train_loader):
            data,mask_t= (data).to(self.device), mask_t.to(self.device)
            data=data*1

            for Epoch in tqdm(range(Epochs)):
                self.generator.train()
                
                M,img,MagnetisationTheta,MagnetisationPhi,NVtheta,NVphi = self.generator(data,mask_t,PositiveMagnetisationOnly=PositiveMagnetisationOnly,IntegerOnly=False, MagnetisationLayerRange=MagnetisationLayerRange)
                
                reconstructed_Mx = img[0,0,:,:]
                reconstructed_My = img[0,1,:,:]

                self.PROP.define_magnetisation_transformation()
                d_matrix = torch.from_numpy(self.PROP.d_matrix).to(self.device)

                # Define the transoformation for bx, by, and bz from Mx
                mx_to_bx = d_matrix[0, 0, ::] 
                mx_to_by = d_matrix[0, 1, ::]
                mx_to_bz = d_matrix[0, 2, ::] 
                
                # Remove any nans or ints in the transformation
                tmp_mx_to_bx= torch.logical_or(torch.isnan(mx_to_bx), torch.isinf(mx_to_bx)).to(self.device)
                tmp_mx_to_by= torch.logical_or(torch.isnan(mx_to_by), torch.isinf(mx_to_by)).to(self.device)
                tmp_mx_to_bz= torch.logical_or(torch.isnan(mx_to_bz), torch.isinf(mx_to_bz)).to(self.device)

                mx_to_bx[tmp_mx_to_bx] = 0
                mx_to_by[tmp_mx_to_by] = 0
                mx_to_bz[tmp_mx_to_bz] = 0
                
                
                # Define the transoformation for bx, by, and bz from My
                my_to_bx = d_matrix[1, 0, ::] 
                my_to_by = d_matrix[1, 1, ::] 
                my_to_bz = d_matrix[1, 2, ::] 
                
                # Remove any nans or ints in the transformation
                tmp_my_to_bx= torch.logical_or(torch.isnan(my_to_bx), torch.isinf(my_to_bx)).to(self.device)
                tmp_my_to_by= torch.logical_or(torch.isnan(my_to_by), torch.isinf(my_to_by)).to(self.device)
                tmp_my_to_bz= torch.logical_or(torch.isnan(my_to_bz), torch.isinf(my_to_bz)).to(self.device)

                my_to_bx[tmp_my_to_bx] = 0
                my_to_by[tmp_my_to_by] = 0
                my_to_bz[tmp_my_to_bz] = 0

                #reconstructed_image = torch.zeros(data.shape).to(device)
                fft_magx_image = torch.fft.fft2(reconstructed_Mx).to(self.device)
                fft_magy_image = torch.fft.fft2(reconstructed_My).to(self.device)

                bx = torch.fft.ifft2(mx_to_bx * fft_magx_image).real.to(self.device) + torch.fft.ifft2(my_to_bx * fft_magy_image).real.to(self.device)
                by = torch.fft.ifft2(mx_to_by * fft_magx_image).real.to(self.device) + torch.fft.ifft2(my_to_by * fft_magy_image).real.to(self.device)
                bz = torch.fft.ifft2(mx_to_bz * fft_magx_image).real.to(self.device) + torch.fft.ifft2(my_to_bz * fft_magy_image).real.to(self.device) 


                # Get the NV angle or try to fit the NV angles
                if self.PROP.options['NV']['FindTheta']:
                    NVTheta = torch.deg2rad(NVtheta)
                else:
                    NVtheta = torch.from_numpy(np.array(self.PROP.options['NV']['Theta']))
                    NVTheta = torch.deg2rad(NVtheta)

                if self.PROP.options['NV']['FindPhi']:
                    NVPhi = torch.deg2rad(NVphi)
                else:
                    NVphi = torch.from_numpy(np.array(self.PROP.options['NV']['Phi']))
                    NVPhi = torch.deg2rad(NVphi)

                # Define the NV Orientation
                u_prop=  [torch.sin(NVTheta) * torch.cos(NVPhi), torch.sin(NVTheta) * torch.sin(NVPhi), torch.cos(NVTheta)]

                if  self.PROP.options['Magnetisation']['FindTheta'] or self.PROP.options['Magnetisation']['FindPhi']:
                    bnv =  u_prop[2]*bz
                else:
                    bnv =  u_prop[0]*bx +  u_prop[1]*by + u_prop[2]*bz

                if LossFunction == "L1":
                    G_loss= Loss(data, bnv)
                elif LossFunction == "L2":
                    G_loss =torch.mean((data - bnv)**2)

                SNR = (torch.max(reconstructed_Mx)-torch.min(reconstructed_Mx))/(torch.absolute(torch.std(reconstructed_Mx)))

                # Extract the error function into a list for plotting and returning from the training.
                # Both the L1 and L2 error functions are converted into units of Tesla
                if LossFunction == "L1":
                    G_loss_List.append(G_loss.item()*data.size(0)/unit_conversion)
                    snr_List.append(SNR.item()*data.size(0)/unit_conversion)
                elif LossFunction == "L2":
                    # Take the sqrt to make the output in units of Tesla
                    G_loss_List.append( np.sqrt(G_loss.item()*data.size(0))/ unit_conversion)
                    snr_List.append(SNR.item()*data.size(0)/unit_conversion)

                Errors['Loss Function'] = G_loss_List
                Errors['SNR'] = snr_List

                if Magnetization is None:
                    if PrintLossValue:
                        print('loss = ' +  str(G_loss.item()*data.size(0)))

                # If a magnetization image is given calculate a loss function between the NN and the given image.
                # This is just for comparison but in principle could be used as the NN loss function
                if Magnetization is not None:
                    M_loss = Loss(Magnetization, reconstructed_Mx)
                    if PrintLossValue:
                        print('loss = ' +  str(M_loss.item()*data.size(0)))

                    running_loss =+ M_loss.item()*data.size(0)
                    loss_values.append(running_loss)
                    Errors['Magnetisation Loss'] = loss_values

                self.generator.optimizer.zero_grad()
                G_loss.backward()
                #reconstructed.backward(grads['reconstructed_M'])
                self.generator.optimizer.step()

        # Plot the evolution of the loss function at the end of the train
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(12)
        plt.subplot(1,2,1)
        plt.plot(G_loss_List, label='Loss function')
        #plt.plot(snr_List, label='Loss function')
        plt.ylabel('Average difference, $\Delta B (T)$')
        plt.title('Error function evolution')
        plt.xlabel('Epochs')
        ax.legend()
        plt.legend
        plt.subplot(1,2,2)
        plt.plot(snr_List, label='Loss function')
        plt.ylabel('Standard divation, $\mu (T)$')
        plt.title('Error function evolution')
        plt.xlabel('Epochs')
        ax.legend()
        plt.legend


        NVTheta = torch.rad2deg(NVTheta)
        NVPhi = torch.rad2deg(NVPhi)
        # FOR AFTER UPGRADING TO A CLASS

        # Detatch the results and return them as properties of the class.
        self.MMag = M.detach().cpu().numpy()
        self.NVtheta = NVTheta.detach().cpu().numpy()
        self.NVphi = NVPhi.detach().cpu().numpy()
        self.TrainingErrors = Errors
    
        self.ReconMagX = reconstructed_Mx.detach().cpu().numpy()
        self.ReconMagY = reconstructed_My.detach().cpu().numpy()
        self.ReconBnv = bnv.detach().cpu().numpy()
        
        return bnv, Errors, NVtheta, NVphi

    def extract_multiple_results(self, plotResults = True):

        '''
        This function is used to return the results of the training done in train_cnn
        inputs:
            device: The cpu to gpu device used for the training
            img_input:
            mask_t:
            G_A: The generator for the neural network
            PROP: The class instance for propagation
            b_cnn: The return values from the training function train_cnn
            plotResults (default = True_: Whether or not to plot the results from the machine learning

        '''
        
        # Unit converse is to take the B map from the NN and convert it back into units of Tesla
        unit_conversion_M_to_B =9.27e-24 / 1e-18

        Results = dict()
        MagnetisationXMap = self.ReconMagX
        MagnetisationYMap = self.ReconMagY
        ReconstructedBnv = self.ReconBnv

        Results["Errors"] = self.TrainingErrors
        Results["M Mag"] = self.MMag
        Results["NV theta"]= self.NVtheta
        Results["NV phi"]= self.NVphi


        # When applying the OriginalROI the data gets rotated. The np.rot90 function is there to fix this issue.
        Results["Original B"] = np.rot90(self.PROP.MagneticFieldExtended[self.PROP.OriginalROI] * unit_conversion_M_to_B, k=1)
        Results["Reconstructed B"] = np.rot90(ReconstructedBnv[self.PROP.OriginalROI] * unit_conversion_M_to_B , k=1)
        Results["Magnetisation X"] = np.rot90(MagnetisationXMap[self.PROP.OriginalROI] , k=1)
        Results["Magnetisation Y"] = np.rot90(MagnetisationYMap[self.PROP.OriginalROI] , k=1)
        Results["Magnetisation Mag"] = np.sqrt(Results["Magnetisation X"]**2 + Results["Magnetisation Y"]**2)
        Results["Magnetisation Angle"] = 180/np.pi *np.arctan(np.sqrt(Results["Magnetisation Y"]/Results["Magnetisation X"])) 
                           
        correct_neg_values = np.where(Results["Magnetisation X"] < 0)
        Results["Magnetisation Angle"][correct_neg_values] = Results["Magnetisation Angle"][correct_neg_values] - 180    
            
        #check = np.where(Results["Magnetisation X"] == 0)
        #Results["Magnetisation Angle"][check] = 0     
        
        #check = np.where(Results["Magnetisation Y"] == 0)
        #Results["Magnetisation Angle"][check] = 90  
        
        if plotResults:
            fig = plt.figure()
            fig.set_figheight(12)
            fig.set_figwidth(24)

            ax1 = plt.subplot(2,3,1)

            Range = np.max(np.abs(1e6*Results["Original B"]))
            plt.imshow(1e6*Results["Original B"], cmap="seismic", vmin = -Range, vmax = Range)
            plt.colorbar(fraction=0.046, pad=0.04,label="B  ($\mu T)$")
            scalebar = ScaleBar(self.PROP.options['PixelSize'], location='lower left')
            ax1.add_artist(scalebar)
            plt.xticks([])
            plt.yticks([])
            plt.title('Magnetic Field')

            ax2 = plt.subplot(2,3,2)
            Range = np.max(np.abs(1e6*Results["Reconstructed B"]))
            plt.imshow(1e6*Results["Reconstructed B"], cmap="seismic", vmin = -Range, vmax = Range)
            scalebar = ScaleBar(self.PROP.options['PixelSize'], location='lower left')
            ax2.add_artist(scalebar)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(fraction=0.046, pad=0.04,label="B ($\mu T)$")
            plt.title('reconstructed magneticfield')

            ax3 = plt.subplot(2,3,3)
            PlotData = 1e6*(Results["Original B"] - Results["Reconstructed B"])
            Range = np.max(np.abs(PlotData))
            plt.imshow(PlotData, cmap="seismic", vmin = -Range, vmax = Range)
            scalebar = ScaleBar(self.PROP.options['PixelSize'], location='lower left')
            ax3.add_artist(scalebar)
            plt.xticks([])
            plt.yticks([])
            # plt.clim([-150,150])
            plt.colorbar(fraction=0.046, pad=0.04,label="B  ($\mu T)$")
            plt.title('reconstruction difference B')

            ax4 = plt.subplot(2,3,4)
            PlotData = Results["Magnetisation Mag"]
            Range = np.max(np.abs(PlotData))
            #plt.imshow(PlotData, cmap="PuOr", vmin = -Range, vmax = Range)
            plt.imshow(PlotData)
            scalebar = ScaleBar(self.PROP.options['PixelSize'], location='lower left')
            ax4.add_artist(scalebar)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(fraction=0.046, pad=0.04,label="M  ($\mu_B nm^2$)")
            plt.title('Reconstructed magnetisation magnitude')
            
            ax5 = plt.subplot(2,3,5)
            PlotData = Results["Magnetisation Angle"]
            Range = np.max(np.abs(PlotData))
            #plt.imshow(PlotData, cmap="PuOr", vmin = -Range, vmax = Range)
            plt.imshow(PlotData)
            scalebar = ScaleBar(self.PROP.options['PixelSize'], location='lower left')
            ax4.add_artist(scalebar)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(fraction=0.046, pad=0.04,label="M  ($\mu_B nm^2$)")
            plt.title('Reconstructed magnetisation angle')

        return Results

    def evaluateCNN(self):
        means,stds,pred,ci_upper,ci_lower, ic_acc,ic_acc2,loss = evaluate(self.generator, self.img_input.to(self.device), self.mask_t.to(self.device))
        return  means,stds,pred,ci_upper,ci_lower, ic_acc,ic_acc2,loss

    def train_J(self,   
                  mlp=False, 
                  LossFunction = 'L1',
                  Epochs = 100,
                  PositiveMagnetisationOnly = False,
                  IntegerOnly = False,
                  IntergerInitialTraining = 100,
                  Magnetization = None,
                  PrintLossValue = False,
                  MagnetisationLayerRange = None):

        # training for the current density reconstruction

        ImageSize = self.PROP.options['ImageShape']
        L1_Loss = nn.L1Loss()
        L2_Loss = nn.MSELoss()

        if LossFunction == "L1":
            Loss = L1_Loss
        elif LossFunction == "L2":
            Loss = L2_Loss

        unit_conversion = 1e-18 / 9.27e-24

        running_loss = 0.0
        loss_values = []
        G_loss_List = []
        snr_List = []
        Errors = dict()

        NVtheta = np.deg2rad(0)
        NVphi = np.deg2rad(0)
        u_prop =  [np.sin(NVtheta) * np.cos(NVphi), np.sin(NVtheta) * np.sin(NVphi), np.cos(NVtheta)]

        for epoch in range(100):
            for batch_idx, (data,mask_t) in enumerate(self.train_loader):
                data,mask_t= (data).to(self.device), mask_t.to(self.device)
                data=data*1
                self.generator.train()
                
                M,img,MagnetisationTheta,MagnetisationPhi,NVtheta,NVphi = self.generator(data,mask_t,PositiveMagnetisationOnly=PositiveMagnetisationOnly,IntegerOnly=False, MagnetisationLayerRange=MagnetisationLayerRange)
                
                curr1 = img[0,0,:,:]
                curr2 = img[0,1,:,:]

                '''
                data,clean= data.to(self.device), clean.to(self.device)
                self.generator.train()
                curr1, curr2= self.generator(data,clean)
                '''

                fft_jx_image = (torch.fft.fft2(curr1)).to(self.device)
                fft_jy_image = (torch.fft.fft2(curr2)).to(self.device)

                self.PROP.define_current_transformation()

                self.PROP.get_image_filter()
                img_filter = self.PROP.img_filter

                bx_to_jx = torch.from_numpy(self.PROP.bx_to_jx).to(self.device)
                bx_to_jy = torch.from_numpy(self.PROP.bx_to_jy).to(self.device)
                by_to_jx = torch.from_numpy(self.PROP.by_to_jx).to(self.device)
                by_to_jy = torch.from_numpy(self.PROP.by_to_jy).to(self.device)
                bz_to_jx = torch.from_numpy(self.PROP.bz_to_jx).to(self.device)
                bz_to_jy = torch.from_numpy(self.PROP.bz_to_jy).to(self.device)

                jx_to_bx=img_filter/bx_to_jx
                jy_to_bx=img_filter/bx_to_jy
                jx_to_by=img_filter/by_to_jx
                jy_to_by=img_filter/by_to_jy
                jx_to_bz=img_filter/bz_to_jx
                jy_to_bz=img_filter/bz_to_jy

                tmp_jx_to_bx= torch.logical_or(torch.isnan(jx_to_bx), torch.isinf(jx_to_bx)).to(self.device)
                tmp_jy_to_bx= torch.logical_or(torch.isnan(jy_to_bx), torch.isinf(jy_to_bx)).to(self.device)
                tmp_jx_to_by= torch.logical_or(torch.isnan(jx_to_by), torch.isinf(jx_to_by)).to(self.device)
                tmp_jy_to_by= torch.logical_or(torch.isnan(jy_to_by), torch.isinf(jy_to_by)).to(self.device)
                tmp_jx_to_bz= torch.logical_or(torch.isnan(jx_to_bz), torch.isinf(jx_to_bz)).to(self.device)
                tmp_jy_to_bz= torch.logical_or(torch.isnan(jy_to_bz), torch.isinf(jy_to_bz)).to(self.device)

                jx_to_bx[tmp_jx_to_bx] = 0
                jy_to_bx[tmp_jy_to_bx] = 0
                jx_to_by[tmp_jx_to_by] = 0
                jy_to_by[tmp_jy_to_by] = 0
                jx_to_bz[tmp_jx_to_bz] = 0
                jy_to_bz[tmp_jy_to_bz] = 0

                a= np.fft.ifft2(jx_to_bx).real

                bx = torch.fft.ifft2(jx_to_bx*fft_jx_image).real + torch.fft.ifft2(jy_to_bx*fft_jy_image).real 
                by = torch.fft.ifft2(jx_to_by*fft_jx_image).real + torch.fft.ifft2(jy_to_by*fft_jy_image).real
                bz = torch.fft.ifft2(jx_to_bz*fft_jx_image).real + torch.fft.ifft2(jy_to_bz*fft_jy_image).real

                bnv =  (u_prop[0]*bx +  u_prop[1]*by + u_prop[2]*bz)*unit_conversion

                G_loss= L1_Loss(data,bnv)

                self.generator.optimizer.zero_grad()
                G_loss.backward()
                self.generator.optimizer.step()

                self.MagTheta = self.PROP.options['Magnetisation']['Theta']
                self.MagPhi = self.PROP.options['Magnetisation']['Theta']
                
                self.NVtheta = NVtheta
                self.NVphi = NVphi
                self.TrainingErrors = Errors

                self.MMag = None
                G_loss_List.append(G_loss.item()*data.size(0)/unit_conversion)

                Errors['Loss Function'] = G_loss_List
                Errors['SNR'] = snr_List

        self.ReconCurrenetJx = curr1.detach().cpu().numpy()
        self.ReconCurrenetJy = curr2.detach().cpu().numpy()
        self.ReconBnv = bnv.detach().cpu().numpy()

        return bnv, Errors
