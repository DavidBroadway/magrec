"""
Module docstring here

"""

__author__ = "Adrien Dubois and David Broadway"

import torch
import torch.nn as nn
import scipy.signal
from torch.utils.data import Dataset, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import json

from Propagator import Propagator
from Generator import generator_CNN, generator_MLP_bayesian
from Train import train_cnn
from  Evaluate import evaluate
import matplotlib as plt

# ============================================================================


def Magnetization(data):

  #f = open('/content/drive/MyDrive/Colab Notebooks/wp2/Blob_simulation_out_t_clean21')
  unit_conversion = 1e-18 / 9.27e-24
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # Extract the data
  #MagneticField = np.asarray(data['ExperimentMagneticField']['BNV']['Data'])
  #MagneticField = np.asarray(data['MagnetisationSimulation']['MagnetisationSimulation']['BNV'])
  #Magnetization = np.array(data['MagnetisationSimulation']['MagnetisationSimulation']['SimMagnetisation'])
  #MagneticField = MagneticField.T - (MagneticField[:,0]).T
  MagneticField = ((data - data.mean()))/unit_conversion

  #display(data['ExperimentMagneticField'].keys())

  # Define the dictionary for the forward propagation


  PropagationOptions = dict()
  PropagationOptions['PixelSize'] =  50e-08
  PropagationOptions['ImageShape'] = 128
  PropagationOptions['NV'] = dict()
  PropagationOptions['NV']['FindTheta']=False
  PropagationOptions['NV']['Theta'] =45
  PropagationOptions['NV']['FindPhi']=False
  PropagationOptions['NV']['Phi'] = 0
  PropagationOptions['NV']['Height'] = 50e-08
  PropagationOptions['use_stand_off'] = True
  PropagationOptions['Magnetisation'] = dict()
  PropagationOptions['Magnetisation']['FindTheta']=False
  PropagationOptions['Magnetisation']['Theta'] = 0
  PropagationOptions['Magnetisation']['FindPhi']=False
  PropagationOptions['Magnetisation']['Phi'] = 20
  PropagationOptions['Mag_z'] = True
  PropagationOptions['unv'] = [0,0,1]
  PropagationOptions['FFT'] = dict()
  PropagationOptions["FFT"]["PaddingFactor"]= None
  PropagationOptions["FFT"]["performPadding"]= None
  PropagationOptions["in_plane_propagation"]= True
  PropagationOptions["in_plane_angle"]= 0
  PROP = Propagator(PropagationOptions, MagneticField,PropagationOptions['ImageShape'])

  PROP.reshape(PropagationOptions['ImageShape'])
  mask = np.where(PROP.MagneticFieldExtended == 0,0,1)
  mask_t=torch.FloatTensor(mask[np.newaxis,np.newaxis])

  img2 = np.where((PROP.MagneticFieldExtended<0.20)&(PROP.MagneticFieldExtended>-0.20),0,PROP.MagneticFieldExtended)
  C = torch.FloatTensor((PROP.MagneticFieldExtended)[np.newaxis,np.newaxis,:,:])
  train_data_3 = TensorDataset(C, mask_t,C)
  train_loader_3 = DataLoader( dataset=(train_data_3))

  ML_options = dict()
  ML_options['FindNVOrientation'] = False
  ML_options['mlp']=False
  ML_options['size']=PropagationOptions['ImageShape']


  G_cnn = generator_CNN(Size=2,ImageSize=PropagationOptions['ImageShape']).to(device)
  G_cnn_optimizer = torch.optim.Adam(G_cnn.parameters())


  res=train_cnn(device, G_cnn, G_cnn_optimizer, train_loader_3, 50, PROP, ML_options)

  means,stds,pred,ci_upper,ci_lower, ic_acc,ic_acc2,loss = evaluate(G_cnn, C.to(device),mask_t.to(device))

  img = G_cnn(C.to(device),mask_t.to(device))
  res=res[0,0].detach().cpu().numpy()
  MagnetisationMap = img[0][0,0,:,:].detach().cpu().numpy()

  border1=int((PropagationOptions['ImageShape'] -MagneticField.shape[0])/2)
  border2=int((PropagationOptions['ImageShape'] -MagneticField.shape[0])/2+MagneticField.shape[0])
  border3=int((PropagationOptions['ImageShape'] -MagneticField.shape[1])/2)
  border4=int((PropagationOptions['ImageShape'] -MagneticField.shape[1])/2+MagneticField.shape[1])

  res2 = res[border1:border2,border3:border4]
  MagnetisationMap2=MagnetisationMap[border1:border2,border3:border4]

  stacked = [res2,MagnetisationMap2]

  return stacked