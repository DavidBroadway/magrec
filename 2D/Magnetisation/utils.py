"""
Module that load the data

"""

__author__ = "Adrien Dubois and David Broadway"

import json
import numpy as np
import matplotlib.pyplot as plt
from Magnetisation.Propagator import Propagator



def LoadData(DataPath=None, Magneticfield=None, ImageShape = 128, Normalise=False, Display=False):

    unit_conversion = 1e-18 / 9.27e-24

    file = open(DataPath, "r")
    data = json.load(file)
    MagneticField = np.asarray(data['ExperimentMagneticField']['BNV']['Data'])


    if DataPath:
        MagneticField = np.asarray(data['ExperimentMagneticField']['BNV']['Data'])

    if Magneticfield:
        MagneticField = Magneticfield

    PropagationOptions = dict()

    if "PixelSizeX" in data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']:
        PropagationOptions['PixelSize'] = data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['PixelSizeX']
        #print(PropagationOptions['PixelSize'])
    else:
        PropagationOptions['PixelSize'] = None
    
    PropagationOptions['ImageShape'] = ImageShape

    PropagationOptions['NV'] = dict()
    if "Height" in data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['BNV']:
        PropagationOptions['NV']['Height'] = data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['BNV']['Height']
        
    else:
        PropagationOptions['NV']['Height'] = None

    if "Theta" in data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['BNV']:
        PropagationOptions['NV']['Theta'] = data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['BNV']['Theta']
        PropagationOptions['NV']['FindTheta'] = False
    else:
        PropagationOptions['NV']['Theta'] = None
        PropagationOptions['NV']['FindTheta'] = True
    
    if "Phi" in data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['BNV']:
        PropagationOptions['NV']['Phi'] = data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['BNV']['Phi']
        PropagationOptions['NV']['FindPhi']= False
    else:
        PropagationOptions['NV']['Phi'] = None
        PropagationOptions['NV']['FindPhi']= True

    
    PropagationOptions['Magnetisation'] = dict()

    if "Phi" in data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['BNV']:
        PropagationOptions['Magnetisation']['Phi'] = data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['Kernal']['AssumedMagDir']
        PropagationOptions['Magnetisation']['FindPhi']= False
    else:
        PropagationOptions['Magnetisation']['Phi'] = None
        PropagationOptions['Magnetisation']['FindPhi']= True

    if "Theta" in data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['BNV']:
        PropagationOptions['Magnetisation']['Theta'] = data['ExperimentMagneticField']['MagnetisationPropagation']['PropStruct']['Kernal']['AssumedMagDir']
        PropagationOptions['Magnetisation']['FindTheta']= False
    else:
        PropagationOptions['Magnetisation']['Theta'] = None
        PropagationOptions['Magnetisation']['FindTheta']= True


    PropagationOptions['FFT'] = dict()
    PropagationOptions["FFT"]["PaddingFactor"]= 2
    PropagationOptions["FFT"]["performPadding"]= False
    PropagationOptions["FFT"]["PaddingMode"] = 'constant'
    PropagationOptions['FFT']['Extended'] = False
    PropagationOptions['FFT']['Extention'] = 100
    PropagationOptions['FFT']['Filter'] = dict()
    PropagationOptions['FFT']['Filter']['useHanning'] = False 
    PropagationOptions['FFT']['Filter']['useHighCutoff'] = False
    PropagationOptions['FFT']['Filter']['useLowCutoff'] = False
    PropagationOptions['FFT']['Filter']['LambdaHighCutoff'] = 100e-9
    PropagationOptions['FFT']['Filter']['LambdaLowCutoff'] = 7e-06  

    if Normalise:
        fft_signal = np.fft.fft2(MagneticField) 
        fft_signal_clean = np.where(fft_signal.imag==0,0,fft_signal)
        MagneticField = np.fft.ifft2(fft_signal_clean).real

    if Display:
        
        fig = plt.figure()
        # to change size of subplot's
        # set height of each subplot as 8
        fig.set_figheight(6)
        # set width of each subplot as 8
        fig.set_figwidth(6)
        #PROP.MagneticFieldExtended = np.where(np.abs(PROP.MagneticFieldExtended) <(PROP.MagneticFieldExtended.mean()+PROP.MagneticFieldExtended.std()), 0, PROP.MagneticFieldExtended)
        DataToPlot = 1e6* MagneticField
        PlotRange = np.max(np.abs(DataToPlot))
        plt.imshow(1e6* MagneticField, cmap="bwr", vmin = -PlotRange, vmax = PlotRange)
        plt.colorbar(label="Magnetic Field (uT)")
        #plt.clim([-20, 20])

    PROP = Propagator(PropagationOptions, MagneticField,PropagationOptions['ImageShape'])

    # Extend the data set to the requested square dimensions
    # Returns to self.MagneticFieldExtended
    PROP.reshape(img=MagneticField)

#     if "Mag" in data['ExperimentMagneticField']['MagnetisationPropagation']['Magnetisation']:
#         img_output = torch.FloatTensor(np.asarray(data['ExperimentMagneticField']['MagnetisationPropagation']['Magnetisation']['Mag'])[np.newaxis,np.newaxis,:,:])
#     else:
#         img_output = Nonet

#     img_input = torch.FloatTensor((PROP.MagneticFieldExtended*unit_conversion)[np.newaxis, np.newaxis, :, :])
    
#     train_data_cnn = TensorDataset(img_input,img_input)
#     train_loader_cnn = DataLoader(dataset=(train_data_cnn))

    return PROP

def SaveDictToJson(FileName, Dictionary):

    # Save the data
    from json import JSONEncoder
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)
    
    print("serialize NumPy array into JSON and write into a file")
    with open(FileName + ".json", "w") as write_file:
        json.dump(Dictionary, write_file, cls=NumpyArrayEncoder)
    print("Done writing serialized NumPy array into file")