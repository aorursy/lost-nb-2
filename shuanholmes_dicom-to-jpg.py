#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import cv2
import shutil
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import scipy.ndimage as ndimage
import scipy
import matplotlib.pyplot as plt
from skimage import measure, morphology, segmentation

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image

# Some of the starting Code is taken from ArnavJain
def generate_markers(image):
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed

def seperate_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    #Watershed algorithm
    '''
    watershed = morphology.watershed(sobel_gradient, marker_watershed)
    
    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    
    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    
    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)))
    
    # return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed
    '''
    return sobel_gradient


# Script starts Here.
INPUT_FOLDER = './stage1'
OUTPUT_FOLDER_1 = './stage1_Sobel_Gradient'
#OUTPUT_FOLDER_2 = "./stage1_Segmented"

if os.path.exists(OUTPUT_FOLDER_1):
	shutil.rmtree(OUTPUT_FOLDER_1)
os.mkdir(OUTPUT_FOLDER_1)

#if os.path.exists(OUTPUT_FOLDER_2):
#	shutil.rmtree(OUTPUT_FOLDER_2)
#os.mkdir(OUTPUT_FOLDER_2)

patients = os.listdir(INPUT_FOLDER)
num = 0
for i in range(len(patients)):
    print 'patient ' + str(num) + ' : ' + patients[i]
    print 'process : ' + str(float(num) / len(patients))
    num += 1

    os.mkdir(OUTPUT_FOLDER_1 + '/' + patients[i])
    #os.mkdir(OUTPUT_FOLDER_2 + '/' + patients[i])
    patient = load_scan(INPUT_FOLDER + '/' + patients[i])
    patient_pixels = get_pixels_hu(patient)
    # pix_resampled = resample(patient_pixels, patient, [1,1,1])

    for j in range(len(patient_pixels)):
		#Some Testcode:
        sobel_gradient = seperate_lungs(patient_pixels[j])
        fig = Figure(figsize=sobel_gradient.shape[::-1], dpi=1, frameon=False)
        canvas = FigureCanvas(fig)
        fig.figimage(sobel_gradient, cmap=plt.cm.gray, vmin=None, vmax=None, origin=None)
        fig.savefig(OUTPUT_FOLDER_1 + '/' + patients[i] + '/' + str(j) + '.jpg', dpi=1, format=None)
        
        #fig = Figure(figsize=segmented.shape[::-1], dpi=1, frameon=False)
        #canvas = FigureCanvas(fig)
        #fig.figimage(segmented, cmap=plt.cm.gray, vmin=None, vmax=None, origin=None)
        #fig.savefig(OUTPUT_FOLDER_2 + '/' + patients[i] + '/' + str(j) + ".jpg", dpi=1, format=None)
        

