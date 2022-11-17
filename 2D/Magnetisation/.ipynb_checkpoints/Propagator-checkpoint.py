

"""
This Module is for propagating from the reconstructed source to the measured fields

"""

__author__ = "Adrien Dubois and David Broadway"

import numpy as np
#from Magnetisation.Evaluate import evaluate

class Propagator(object):
    
    """ Class for dealing with all different types of propagation b xyz, magnetisation, and current
    mapping
    """

    def __init__(self, options, MagneticField=None, Size=None,Magnetisation=None, CurrentDensity_jx=None, CurrentDensity_jy = None):
        self.options = options
        self.MagneticField = MagneticField
        self.Magnetisation = Magnetisation
        self.CurrentDensity_jx = CurrentDensity_jx
        self.CurrentDensity_jy = CurrentDensity_jy

        #self.pixel_size_for_fft = self.options["raw_pixel_size"] * self.options["total_bin"] * 1e-6
        # self.pixel_size_for_fft=50e-9
        # Define the k-vectors for fourier space.
        if Size is not None:
            self.reshape(MagneticField, Size=Size)
            self.kx, self.ky, self.k = self.define_k_vectors(self.MagneticFieldExtended)
        elif Magnetisation is not None:
            self.kx, self.ky, self.k = self.define_k_vectors(Magnetisation)
        elif CurrentDensity_jx is not None:
            self.kx, self.ky, self.k = self.define_k_vectors(CurrentDensity_jx)
        elif MagneticField is not None:
            self.reshape(img=MagneticField, Size=self.options['ImageShape'])
            self.kx, self.ky, self.k = self.define_k_vectors(self.MagneticFieldExtended)
        #else:
            #self.kx, self.ky, self.k = self.define_k_vectors()

    def reshape(self,img=None, Size=None):
        if img is None:
            MagnetisationDataList = self.MagneticField
        else:
            MagnetisationDataList = img
        
        print(img)
        image_size = img.shape

        if(MagnetisationDataList.shape[0]>MagnetisationDataList.shape[1]):
            max=MagnetisationDataList.shape[0]
        else:
            max=MagnetisationDataList.shape[1]

        n=2
        while(max>n):
            n=n*2
        Size=n
        
        if MagnetisationDataList.shape[0]<(Size+1):
            A = np.int(Size - MagnetisationDataList.shape[0])
            if (A%2==0):
                #tmp=int(A/2)
                MagnetisationDataList = np.pad(MagnetisationDataList, ((int(A/2),int(A/2)), (0, 0)), 'constant', constant_values=0)
            else:
                MagnetisationDataList = np.pad(MagnetisationDataList, ((int((A-1)/2),int((A+1)/2)), (0, 0)), 'constant', constant_values=0)

        else:
            MagnetisationDataList = MagnetisationDataList[:Size,:Size]

        if MagnetisationDataList.shape[1]<(Size+1):

            A = Size - MagnetisationDataList.shape[1]
            if (A%2==0):
                MagnetisationDataList = np.pad(MagnetisationDataList, ((0, 0), (int(A/2),int(A/2))), 'constant',constant_values=0)
            else:
                MagnetisationDataList = np.pad(MagnetisationDataList,
                                             ((0, 0), (int((A-1)/2),int((A+1)/2))), 'constant', constant_values=0)
        else:
             MagnetisationDataList = MagnetisationDataList[:Size,:Size]
        unit_conversion = 1e-18 / 9.27e-24
        self.MagneticFieldExtended = MagnetisationDataList*unit_conversion
        # Define the region of interest for plotting
        padded_shape = self.MagneticFieldExtended.shape
        centre = [padded_shape[0] // 2, padded_shape[1] // 2]
        x = [
          np.linspace(
              centre[0] + (image_size[0]-1) / 2,
              centre[0] -  (image_size[0]-1) / 2,
              image_size[0]-1,
              dtype=int,
          )
        ]
        y = [
          np.linspace(
              centre[1] -  (image_size[1]-1) / 2,
              centre[1] + (image_size[1]-1) / 2,
              image_size[1]-1,
              dtype=int,
          )
        ]
        self.Mask = np.where(self.MagneticFieldExtended == 0,0,1)
        self.OriginalROI = np.meshgrid(x, y)

    # =================================

    def define_k_vectors(self, Image = None):

        # scaling for the k vectors so they are in the right units
        if Image is None:
            image_shape = self.options['ImageShape']
        else:
            image_shape = np.shape(Image)

        scalingX = np.float64(2 * np.pi / self.options['PixelSize'])
        
        if 'PixelSizeY' in self.options:
            scalingY = np.float64(2 * np.pi / self.options['PixelSizeY'])
        else:
            scalingY = scalingX
        
        use_k_vector_epsilon=False
        k_vector_epsilon=1e-20
        # get the fft frequencies and forces type to be float64
        kx_vec = scalingX * np.fft.fftfreq(image_shape[1])
        ky_vec = scalingY * np.fft.fftfreq(image_shape[0])
        
        # Include a small factor in the k vectors to remove division by zero issues (min_k)
        # Make a meshgrid to pass back
        if use_k_vector_epsilon:
            kx, ky = np.meshgrid(
                kx_vec + k_vector_epsilon, ky_vec + k_vector_epsilon
            )
        else:
            kx, ky = np.meshgrid(kx_vec, ky_vec)
            
        #kx[1,1] = 0;
        #ky[1,1] = 0;
        # Define the k mag vector
        k = np.sqrt(kx ** 2 + ky ** 2)
        # Take the negative of ky to maintain the correct image rotation
        return kx, ky, k

    # =================================

    def b_xyz(self):
        """ Propagates the bnv data to bxyz.
        Parameters:
            bnv (2D array, float): image/map of the Bnv data.

            nv_axis (list, float): the unv orientation e.g. [0.57, 0, 0.87]

        Returns:
            bx_region, by_region, bz_region (2D arrays, float): images/maps of the bxyz magnetic
            fields after propagation.
        """
        # pad the image for better FFT
        padded_bnv, padding_roi = self.pad_image(self.MagneticField)
        # Perform the FFT
        fft_bnv = np.fft.fft2(padded_bnv)

        # get the transformations
        bnv2bx, bnv2by, bnv2bz = self.define_b_xyz_transformation()
        # ==== Define filter ==== #
        img_filter = self.get_image_filter()
        # Define all of the inverse transformation matrices
        bnv2bx = self.remove_invalid_elements(self.img_filter * bnv2bx)
        bnv2by = self.remove_invalid_elements(self.img_filter * bnv2by)
        bnv2bz = self.remove_invalid_elements(self.img_filter * bnv2bz)
        # transform into xyz
        fft_bx = fft_bnv * bnv2bx
        fft_by = fft_bnv * bnv2by
        fft_bz = fft_bnv * bnv2bz
        # remove the DC component as it is lost in the FFT anyway
        # fft_bx[self.k < self.min_k] = 0
        # fft_by[self.k < self.min_k] = 0
        # fft_bz[self.k < self.min_k] = 0
        # fourier transform back into real space
        bx = np.fft.ifft2(fft_bx).real
        by = np.fft.ifft2(fft_by).real
        bz = np.fft.ifft2(fft_bz).real
        # Readout the non padded region
        if self.options['FFT']["performPadding"]:
            bx_region = bx[padding_roi[0], padding_roi[1]]
            by_region = by[padding_roi[0], padding_roi[1]]
            bz_region = bz[padding_roi[0], padding_roi[1]]
        else:
            bx_region = bx
            by_region = by
            bz_region = bz
        return bx_region, by_region, bz_region

    # =================================

    def bxyz_from_mag(self, Magnetisation):
        """
        Deals with the transformation of a mag map into both bxyz maps.
        """

        NVParams = self.options['NV']
        #MagParams = self.options['Magnetisation']
        # === Perform the FFT === #
        fft_mag_image = np.fft.fft2(Magnetisation)

        # === Define the magnetisation direction ==== #
        MagTheta = np.deg2rad(self.options['Magnetisation']['Theta'])
        MagPhi = np.deg2rad(self.options['Magnetisation']['Phi'])

        # ==== define transformation matrix ==== #
        self.define_magnetisation_transformation()

        d_matrix = self.d_matrix

        # Transformation
        m_to_bx = (np.cos(MagPhi) * np.sin(MagTheta) * d_matrix[0, 0, ::] + np.sin(MagPhi) * np.sin(MagTheta) * d_matrix[1, 0, ::] + np.cos(MagTheta) * d_matrix[2, 0, ::] )
        m_to_by = (np.cos(MagPhi)* np.sin(MagTheta)  * d_matrix[0, 1, ::] + np.sin(MagPhi) * np.sin(MagTheta) * d_matrix[1, 1, ::] +  np.cos(MagTheta) * d_matrix[2, 1, ::])
        m_to_bz = (np.cos(MagPhi) * np.sin(MagTheta) * d_matrix[0, 2, ::] + np.sin(MagPhi) * np.sin(MagTheta) * d_matrix[1, 2, ::] +  np.cos(MagTheta) * d_matrix[2, 2, ::])

        # No filter is required for the propagation from M to B as it is a forward propagation. Only reverse propagation requires the Hanning filters. 

        # Replace all nans and infs with zero
        m_to_bx = self.remove_invalid_elements(m_to_bx)
        m_to_by = self.remove_invalid_elements(m_to_by)
        m_to_bz = self.remove_invalid_elements(m_to_bz)

        bx = np.fft.ifft2(m_to_bx * fft_mag_image).real
        by = np.fft.ifft2(m_to_by * fft_mag_image).real
        bz = np.fft.ifft2(m_to_bz * fft_mag_image).real

        # Define the NV axis for projection
        NVtheta = np.deg2rad(NVParams['Theta'])
        NVPhi = np.deg2rad(NVParams['Phi'])
        u_prop = [np.sin(NVtheta) * np.cos(NVPhi), np.sin(NVtheta) * np.sin(NVPhi) ,np.cos(NVtheta)]

        bnv =  u_prop[0]*bx + u_prop[1]*by + u_prop[2]*bz
        return bnv, m_to_bx, m_to_by, m_to_bz, u_prop, NVtheta, NVPhi

    # =================================

    def bxyz_from_curr(self):
        """
        Deals with the transformation of a mag map into both bxyz maps.
        """
        # === Perform the FFT === #
        fft_jx_image = np.fft.fft2(self.CurrentDensity_jx)
        fft_jy_image = np.fft.fft2(self.CurrentDensity_jy)

        # ==== define transformation matrix ==== #
        self.define_current_transformation()

        # ==== get the image filter ==== #
        self.get_image_filter()
        # Apply filter and Replace all nans and infs with zero
        jx_to_bx = self.remove_invalid_elements(self.img_filter / self.bx_to_jx)
        jy_to_bx = self.remove_invalid_elements(self.img_filter / self.bx_to_jy)
        jx_to_by = self.remove_invalid_elements(self.img_filter / self.by_to_jx)
        jy_to_by = self.remove_invalid_elements(self.img_filter / self.by_to_jy)
        jx_to_bz = self.remove_invalid_elements(self.img_filter / self.bz_to_jx)
        jy_to_bz = self.remove_invalid_elements(self.img_filter / self.bz_to_jy)

        bx = np.fft.ifft2(jx_to_bx * fft_jx_image).real + np.fft.ifft2(jy_to_bx * fft_jy_image).real
        by = np.fft.ifft2(jx_to_by * fft_jx_image).real + np.fft.ifft2(jy_to_by * fft_jy_image).real
        bz = np.fft.ifft2(jx_to_bz * fft_jx_image).real+np.fft.ifft2(jy_to_bz * fft_jy_image).real

        # Define the NV axis for projection
        NVtheta = np.deg2rad(self.options['NV']['Theta'])
        NVPhi = np.deg2rad(self.options['NV']['Phi'])
        u_prop = [np.sin(NVtheta) * np.cos(NVPhi), np.sin(NVtheta) * np.sin(NVPhi) ,np.cos(NVtheta)]

        bnv =  u_prop[0]*bx + u_prop[1]*by + u_prop[2]*bz
        return bnv,u_prop ,bz, jx_to_bz,jy_to_bz,fft_jx_image,fft_jy_image,self.CurrentDensity_jx,self.CurrentDensity_jy


        # =================================

    def current(self):
        """ Takes a b_map or the vector b maps (bx, by, bz) and returns the current density
        jx, jy, jnorm) responsible for producing the magnetic field.

        u_proj defines the axis the b map is measured about. This could be a nv axis or standard
        cartesian coords. If no axis is given the assumption is made that you are using a b map from
        an NV axis that corresponds to the furthest split NV calculated from the magnetic field
        parameters.
        """
        
        unit_conversion = 1e-18 / 9.27e-24
        
        b_image = self.MagneticFieldExtended / unit_conversion

        # === pad the image for better FFT === #
        padded_b_image, padding_roi = self.pad_image(b_image)
        # === Perform the FFT === #
        fft_b_image = np.fft.fft2(padded_b_image)
        fft_b_image = self.remove_invalid_elements(fft_b_image)

        # ==== define transformation matrix ==== #
        self.define_current_transformation()


        # ==== get the image filter ==== #
        self.get_image_filter()
        # Apply filter and Replace all nans and infs with zero
        bx_to_jx = self.remove_invalid_elements(self.img_filter * self.bx_to_jx)
        bx_to_jy = self.remove_invalid_elements(self.img_filter * self.bx_to_jy)
        by_to_jx = self.remove_invalid_elements(self.img_filter * self.by_to_jx)
        by_to_jy = self.remove_invalid_elements(self.img_filter * self.by_to_jy)
        bz_to_jx = self.remove_invalid_elements(self.img_filter * self.bz_to_jx)
        bz_to_jy = self.remove_invalid_elements(self.img_filter * self.bz_to_jy)


        # Define the NV axis for projection
        NVtheta = np.deg2rad(self.options['NV']['Theta'])
        NVPhi = np.deg2rad(self.options['NV']['Phi'])
        u_prop = [np.sin(NVtheta) * np.cos(NVPhi), np.sin(NVtheta) * np.sin(NVPhi) ,np.cos(NVtheta)]

        fft_jx =  u_prop[0]*bx_to_jx * fft_b_image +  + u_prop[1] * by_to_jx * fft_b_image + u_prop[2]* bz_to_jx * fft_b_image
        fft_jy =  u_prop[0]*bx_to_jy * fft_b_image +  + u_prop[1] * by_to_jy * fft_b_image + u_prop[2]* bz_to_jy * fft_b_image
        
       
        
        # ==== inverse FFT ==== #
        jx = np.fft.ifft2(fft_jx).real 
        jy = np.fft.ifft2(fft_jy).real
        # Readout the non padded region
        if self.options['FFT']['performPadding']:
            jx = jx[padding_roi[0], padding_roi[1]].T
            jy = jy[padding_roi[0], padding_roi[1]].T

        # convert to Amp/ micron
        j_norm = np.sqrt(jx ** 2 + jy ** 2)
        return jx, jy, j_norm



    # =================================
    # Define the transformations
    # =================================

    def define_magnetisation_transformation(self):
        """
        Defines the transformation matrix that takes magnetisation to b such that
        b_xyz = d_matrix . m_xyz
        """
        kx= self.kx
        ky = self.ky
        k = self.k
        mu0 = 4 * np.pi * 1e-7
        exp_factor = (mu0/2) * np.exp(- k * self.options['NV']['Height'])

        # Definition of the transformation matrix
        self.d_matrix = (
           exp_factor
          * np.array(
              [
                  [-(kx ** 2) / k, -(kx * ky) / k, -1j * kx],
                  [-kx * ky / k, -(ky ** 2 / k), -1j * ky],
                  [-1j * kx, -1j * ky, k],
              ]
          )
        )
        return

    # =================================

    def define_current_transformation(self):
        """
        Defines the transformation matrix that takes b to J
        """
        NVParams = self.options['NV']

        exp_factor = np.exp(-1 * self.k * NVParams['Height'])
        mu0 = np.pi * 4e-7
        g = -mu0 / 2 * exp_factor

        self.bx_to_jx = 0* self.ky
        self.bx_to_jy = (0 * self.kx + 1) / g

        self.by_to_jx = (0 * self.ky + 1) / g
        self.by_to_jy = 0* self.ky

        self.bz_to_jx = self.ky / (g * (1j * self.k))
        self.bz_to_jy = -1 * self.kx / (g * (1j * self.k))

        return

    # =================================

    def define_b_xyz_transformation(self):
        """
        Defines the transformation that takes bnv to b_xyz
        """
        # Define the NV axis for projection
        NVtheta = np.deg2rad(self.options['NV']['Theta'])
        NVPhi = np.deg2rad(self.options['NV']['Phi'])
        unv = [np.sin(NVtheta) * np.cos(NVPhi), np.sin(NVtheta) * np.sin(NVPhi) ,np.cos(NVtheta)]

        bnv2bx = 1 / (unv[0] + unv[1] * self.ky / self.kx + 1j * unv[2] * self.k / self.kx)
        bnv2by = 1 / (unv[0] * self.kx / self.ky + unv[1] + 1j * unv[2] * self.k / self.ky)
        bnv2bz = 1 / (-1j * unv[0] * self.kx / self.k - 1j * unv[1] * self.ky / self.k + unv[2])
        return bnv2bx, bnv2by, bnv2bz

        # =================================
        # ======= General FFT funcs =======
        # =================================

    def remove_invalid_elements(self, Image):
        """ replaces NaNs and infs with zero"""
        idxs = np.logical_or(np.isnan(Image), np.isinf(Image))
        Image[idxs] = 0
        return Image

    # =================================

    def get_image_filter(self, useHanning = False, useHighCutoff = False, LambdaHighCutoff = 1e9, LambdaLowCutoff=1e-3, useLowCutoff = False):
        """ Computes a hanning image filter with both low and high pass filters.

        Returns:
            img_filter (2d array, float): bandpass filter to remove artifacts in the FFT process.
        """
        
        if 'Filter' in self.options['FFT']:
            # Define Hanning filter to prevent noise amplification at frequencies higher than the
            # spatial resolution
            
            if  useHanning:
                # Hanning filter method

                kkrit = 2*np.pi/ self.options['NV']['Height'];
                self.img_filter  = 0.5*(1 + np.cos(self.k*self.options['NV']['Height']/2));
                self.img_filter [abs(self.k)>kkrit] = 0;

                # apply frequency cutoffs
                if useHighCutoff:
                    k_cut_high = 2 * np.pi / LambdaHighCutoff
                    self.img_filter[(self.k > k_cut_high)] = 0
                if useLowCutoff:
                    k_cut_low = 2 * np.pi / LambdaLowCutoff
                    self.img_filter[(self.k < k_cut_low)] = 0
            else:
                self.img_filter = np.array(1)
        else:
            self.img_filter = np.array(1)
        return

    # =================================

    def pad_image(self, image):
        """
          Apply a padding to the image to prepare for the FFT

          Parameters:
              image (2D array): image to have padding applied too

          Returns:
              padded_image (2D array): image with addition padding if request, otherwise returns
              the original image.

              padding_roi (2D array): is a meshgrid of indices that contain the all of the non
              padded elements. This is used for plotting in different sections of the code.

          """
        # get the shape of the image
        image_size = image.shape
        if self.options['FFT']['performPadding']:
            # define the padding size
            y_pad = self.options["FFT"]["PaddingFactor"]  * image_size[1]
            x_pad = self.options["FFT"]["PaddingFactor"] * image_size[0]

            # < TODO > check why the padding sometimes causes a transpose in the data (maybe a
            #  dict vs list issue).
            # performing the padding
            padded_image = np.pad(
              image,
              mode=self.options["FFT"]["PaddingMode"],
              pad_width=((y_pad // 2, y_pad // 2), (x_pad // 2, x_pad // 2)),
            )
        else:
            padded_image = image
        # Define the region of interest for plotting
        padded_shape = padded_image.shape
        centre = [padded_shape[0] // 2, padded_shape[1] // 2]
        x = [
          np.linspace(
              centre[0] - image_size[0] / 2,
              centre[0] + image_size[0] / 2,
              image_size[0],
              dtype=int,
          )
        ]
        y = [
          np.linspace(
              centre[1] - image_size[1] / 2,
              centre[1] + image_size[1] / 2,
              image_size[1],
              dtype=int,
          )
        ]
        padding_roi = np.meshgrid(x, y)
        return padded_image, padding_roi

    
    def extendData(self, Image):
        # function to extend BNV data with gaussian drop off, over 1/4 of the
        # extension length.

        # Input Parameters
        # BNV (in Tesla) is magnetic field map along BNV
        # extension (in pixel) is number of pixel to extend data

        # Output Parameters
        # extended data

        extension = self.options['FFT']['Extention']
        
        image_size = Image.shape
        
        newSizeX = image_size[0] + 2 * extension
        newSizeY = image_size[1] + 2 * extension
        
        if self.options['FFT']['Extended']:
            extendedData = np.zeros((newSizeX,newSizeY));
            extendedData[extension : -extension, extension : -extension] = Image;

            for i in range(extension):
                extendedData[-extension + i , ::] = np.exp(-(i/(0.25*extension))**2) * extendedData[-extension, ::]

            for i in range(extension):
                extendedData[extension - i , ::] = np.exp(-(i/(0.25*extension))**2) * extendedData[extension, ::]

            for i in range(extension):
                extendedData[:: ,  -extension + i] = np.exp(-(i/(0.25*extension))**2) * extendedData[::, -extension]

            for i in range(extension):
                extendedData[:: , extension - i] = np.exp(-(i/(0.25*extension))**2) * extendedData[::, extension]
            # Define the region of interest for plotting
            padded_shape = extendedData.shape
            centre = [padded_shape[0] // 2, padded_shape[1] // 2]
            x = [
              np.linspace(
                  centre[0] - image_size[0] / 2,
                  centre[0] + image_size[0] / 2,
                  image_size[0],
                  dtype=int,
              )
            ]
            y = [
              np.linspace(
                  centre[1] - image_size[1] / 2,
                  centre[1] + image_size[1] / 2,
                  image_size[1],
                  dtype=int,
              )
            ]
            ROI = np.meshgrid(x, y)
        else:
            extendedData =  Image
            ROI = 1
            
        return extendedData, ROI

    
# =================================
# === Magnetisation propagation ===
# =================================

    def magnetisation(self, b_image=None):
        """ takes a magnetic image and returns the magnetisation that produced it
        """
        NVParams = self.options['NV']
        
        if b_image is None:
            b_image = self.MagneticField

        # === Extend data === #
        b_image_extended, extend_ROI = self.extendData(b_image)
        
        # === pad the image for better FFT === #
        padded_b_image, padding_roi = self.pad_image(b_image_extended)
        # === Perform the FFT === #
        fft_b_image =np.fft.fft2(padded_b_image)
       
        # ==== define k vectors ==== #
        self.kx, self.ky, self.k = self.define_k_vectors(fft_b_image)
        
         # get rid of DC component
        #  fft_b_image[self.k<1e-10] = 0
        
        # ==== define transformation matrix ==== #
        self.define_magnetisation_transformation()
        
        # Define the NV axis for projection
        NVtheta = np.deg2rad(NVParams['Theta'])
        NVPhi = np.deg2rad(NVParams['Phi'])
        u_prop = [np.sin(NVtheta) * np.cos(NVPhi), np.sin(NVtheta) * np.sin(NVPhi) ,np.cos(NVtheta)]
        
        mz_to_b = 1 / (
            u_prop[0] * self.d_matrix[2, 0, ::]
            + u_prop[1] * self.d_matrix[2, 1, ::]
            + u_prop[2] * self.d_matrix[2, 2, ::]
        )
        
        # ==== Define filter ==== #
        self.get_image_filter(**self.options['FFT']['Filter'])

        # Apply filter and define bnv to mz array
        mz_to_b = self.remove_invalid_elements(mz_to_b)
        b_to_mz = self.img_filter *mz_to_b

        # Replace all nans and infs with zero
        b_to_mz = self.remove_invalid_elements(b_to_mz)
        fft_b_image = self.remove_invalid_elements(fft_b_image)
        # remove the DC component as it is lost in the FFT anyway
        fft_b_image[np.abs(self.k) < 1e-10] = 0
        # Transform the bnv into mz
        fft_mz = fft_b_image * b_to_mz


        # ==== inverse FFT ==== #
        mz = np.fft.ifft2(fft_mz).real

        # Readout the non padded region
        if self.options["FFT"]["performPadding"]:
            mz = mz[padding_roi[0], padding_roi[1]]
        if self.options["FFT"]["Extended"]:
            mz = mz[extend_ROI[0],extend_ROI[1]]
        # conversion into more useful units
        # m^2 -> nm^2 = 1e-18
        # A -> uB/m^2 = 9.27e-24
        unit_conversion = 1e-18 / 9.27e-24
        
        return mz * unit_conversion