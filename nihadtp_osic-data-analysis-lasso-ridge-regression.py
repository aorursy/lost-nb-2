#!/usr/bin/env python
# coding: utf-8

# In[19]:


pip install pyspark


# In[20]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)


spark = SparkSession .builder .appName("Test data prediction") .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.enabled", "false")
rawData = spark.read .format('csv') .option('header', 'true') .load('../input/osic-pulmonary-fibrosis-progression/train.csv')

rawData = rawData.select(col('FVC').cast('float'), col('Percent').cast('float'),                         col('Age').cast('float'), col('Sex'), col('SmokingStatus'),                         col('Patient'), col('Weeks').cast('float'))

rawData.toPandas().head()


# In[21]:


from pyspark.ml.stat import Summarizer
from pyspark.ml.feature import VectorAssembler

stat = VectorAssembler(
    inputCols= ['Percent', 'Age', 'FVC', 'Weeks'],
    outputCol= 'feature',
    handleInvalid="keep"
    ).transform(rawData)
summarizer = Summarizer.metrics("mean", "min", "max", "variance")
stat.select(summarizer.summary(stat.feature)).toPandas()


# In[22]:


from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
from pyspark.ml.feature import Normalizer
import seaborn as sns
import pandas as pd

normalizedFeature = Normalizer(inputCol='feature', outputCol='normalized feature').transform(stat)
pearsonCorr = Correlation.corr(normalizedFeature, 'normalized feature').collect()[0][0].toArray()
df = pd.DataFrame(pearsonCorr, columns= ['Percent', 'Age', 'FVC', 'Weeks'] )
sns.heatmap(df, xticklabels= ['Percent', 'Age', 'FVC', 'Weeks'], yticklabels= ['Percent', 'Age', 'FVC', 'Weeks'])

print("From heatmap Age and Percentage is highly positively correlated, Also FVC and Weeks are highly negatively correlated. As Weeks go by FVC keeps on decreasing" )


# In[23]:


#Function to get dicom data
from pydicom.filebase import DicomBytesIO
from pydicom.dataset import Dataset

def getDicomData(binry):
    dicom_bytes = DicomBytesIO(binry)
    return dicom_bytes


# In[24]:


#Function to get and display plotted contours
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pydicom import dcmread


def getLungContours(data, display = False):
   
   try:
       ds = dcmread(data, force = True)
   except Exception:
       return "Unknown", 0, list(np.array([[[0, 0]]]))
   
   try:
       pixel_data = ds.pixel_array
   except RuntimeError:
       pixel_data  = np.ones((512, 512))
       
   img_2d = pixel_data.astype(float)
   img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
   img_2d_scaled = np.uint8(img_2d_scaled)
   kernel = np.ones((3,3),np.uint8)
   gray = img_2d_scaled.copy()

   
   norm_image = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
   
   middle = norm_image[int(512/5):int(512/5*4),int(512/5):int(512/5*4)] 
   mean = np.mean(middle)  
   max = np.max(norm_image)
   min = np.min(norm_image)
   
   norm_image[norm_image==max]=mean
   norm_image[norm_image==min]=mean
   

   # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
   # Set flags (Just to avoid line break in the code)
   flags = cv2.KMEANS_RANDOM_CENTERS
   # Apply KMeans
   z = np.reshape(middle,[np.prod(middle.shape),1])

   compactness,labels,centers = cv2.kmeans(z,                                           2,None,criteria,10,flags)
     
   
   threshold_value = np.mean(centers)

   ret,thresh = cv2.threshold(gray, threshold_value,1,cv2.THRESH_BINARY)

   img_erosion = cv2.erode(thresh, np.ones((3,3),np.uint8), iterations=2) 
   img_erosion = np.uint8(img_erosion * 255)
   invert = 255 - img_erosion

   no_of_labels, output, stats, centroids = cv2.connectedComponentsWithStats(invert, connectivity=8)
   
   good_labels = set()
   
   for label in range(no_of_labels):
       stat = stats[label]
       x_start = stat[0]
       x_end = x_start + stat[2]
       y_start = stat[1]
       y_end = y_start + stat[3]
       
       if x_start > 20 and x_end < 500 and y_start > 50 and y_end < 500:
           good_labels.add(label)        

   for row in range(512):
       for col in range(512):
           if output[row][col] not in good_labels:
               output[row][col] = 0
               
   lungs = np.uint8(73*output/np.max(output))
   lung_fmask = np.uint8(255*output/np.max(output))
   masked_lungs = cv2.bitwise_or(gray, lung_fmask)

   blank_ch = 255 * np.ones_like(lungs)

   hsv = cv2.merge([lungs, blank_ch, blank_ch])

   hsv = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR) 

   contours, hierarchy = cv2.findContours(lung_fmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   cnts_image = cv2.drawContours(hsv.copy(), contours, -1, (0,0,0), 6)
   
   week = str(ds[0x20, 0x13].value)
   patient_id = str(ds[0x10, 0x20].value)
#################DISPLAY###FUNCTIONS################

   if display:
       
       plt.imshow(gray, cmap='Greys')
       plt.title("Grey image " + week)
       plt.show()

       plt.imshow(thresh, cmap='Greys')
       plt.title("Binary Threshold " + week)
       plt.show()

       plt.imshow(img_erosion, cmap='Greys')
       plt.title("erosion " + week)
       plt.show()

       plt.imshow(invert, cmap='Greys')
       plt.title("inversion image" + week)
       plt.show()

       plt.imshow(masked_lungs, cmap='Greys', vmin=0, vmax=255)
       plt.title("Masked Lungs " + week)
       plt.show()

       plt.imshow(hsv, cmap='Greys')
       plt.title("HSV Image of Lungs " + week)
       plt.show()

       plt.imshow(cnts_image, cmap='Greys', vmin=0, vmax=255)
       plt.title("Contours Detected " + week)
       plt.show()
   else:
       return patient_id, int(week), contours


# In[25]:


getLungContours('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/5.dcm', True)


# In[26]:


# Calculating total areas and Total average aspect ratios of all contours in a CT Scan.
#Percentage Ratio of total areas of contours to the total area of canvas

def getContourProperties(contours, display = False):
    
    contour_areas = [cv2.contourArea(cnt) for cnt in contours]

    total_contour_area = sum(contour_areas)
    
    percent_ratio = float(100 * total_contour_area)/(512 * 512)

    aspect_ratios_list = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if h == 0:
            #print("Bounding Rectangle hieght is zero")
            aspect_ratios_list.append(0)
        else:
            
            aspect_ratio = float(w)/h
            aspect_ratios_list.append(aspect_ratio)

    total_aspect_ratios = sum(aspect_ratios_list)
    
    if len(aspect_ratios_list) == 0:
        #print("Aspect Ratio List is zero length")
        avg_aspect_ratio = 0
    else:
        avg_aspect_ratio = total_aspect_ratios/len(aspect_ratios_list)

    if display:
        
        print("Total average aspect ratio -> " + str(avg_aspect_ratio))
        print("Percentage Area Ratio -> " + str(percent_ratio))
        print("Total contour area -> " + str(total_contour_area))
        print("Number of contours detected -> " + str(len(contours)))
        
    else:
        
        return float(avg_aspect_ratio), float(percent_ratio), float(total_contour_area), float(len(contours))


# In[27]:


sample_contour_info = getLungContours('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/5.dcm')
getContourProperties(sample_contour_info[2], True)


# In[ ]:


# Getting CT Scan image's meta data

def get_observation_data(data, display = False):
    int_columns = ["SliceThickness", "KVP", "DistanceSourceToDetector", 
        "DistanceSourceToPatient", "GantryDetectorTilt", "TableHeight", 
        "XRayTubeCurrent", "GeneratorPower", "WindowCenter", "WindowWidth", 
        "SliceLocation", "RescaleIntercept", "RescaleSlope"]
    
    bad_data = {}
    try:
        image_data = dcmread(data, force = True)
    except Exception:
        
        for k in int_columns:
            bad_data[k] = 0.0
        return bad_data
    
#     image_data = dcmread(data, force = True)
    # Dictionary to store the information from the image
    observation_data = {}

    # Integer columns
    
    for k in int_columns:
        if k in image_data:
            try:
                k_value = int(image_data.get(k))
            except TypeError:
                k_value = 0
        else:
            k_value = 0
        observation_data[k] = k_value
    if display:
        
        for i in observation_data:
            print(i + " -> " + str(observation_data[i]))
            
    else:
        
        return observation_data


# In[ ]:


# Get Meta + contour properties 


def getMetaAndContour(data, display = False):
    patient, week, contour = getLungContours(data)
    avg_aspect_ratio, percent_ratio, total_contour_area, no_of_contours = getContourProperties(contour)
    metadata = get_observation_data(data)
    
    output = {
        'Patient': str(patient),
        'Weeks': float(week),
        'AverageAspectRatio': float(avg_aspect_ratio),
        'PercentageRatio': float(percent_ratio),
        'TotalContourArea': float(total_contour_area),
        'NumberOfContours': float(no_of_contours),
        'SliceThickness': float(metadata['SliceThickness']),
        'KVP': float(metadata['KVP']),
        'DistanceSourceToDetector': float(metadata['DistanceSourceToDetector']),
        'DistanceSourceToPatient': float(metadata['DistanceSourceToPatient']),
        'GantryDetectorTilt': float(metadata['GantryDetectorTilt']),
        'TableHeight': float(metadata['TableHeight']),
        'XRayTubeCurrent': float(metadata['XRayTubeCurrent']),
        'GeneratorPower': float(metadata['GeneratorPower']),
        'WindowCenter': float(metadata['WindowCenter']),
        'WindowWidth': float(metadata['WindowWidth']),
        'SliceLocation': float(metadata['SliceLocation']),
        'RescaleIntercept': float(metadata['RescaleIntercept']),
        'RescaleSlope': float(metadata['RescaleSlope'])
    }
    
    if display:
        print(output)
    else:
        return output


# In[ ]:


from pydicom import dcmread
from pydicom.filebase import DicomBytesIO
from pyspark.sql.types import StringType, FloatType, StructType, StructField, MapType, ByteType, Row
from pyspark.sql.functions import udf, col, lit
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# Dicom parser    
def getMap(data):
    binary = getDicomData(data[1])
    dict_ouptut = getMetaAndContour(binary)
    return (dict_ouptut, )
    
rdd = sc.binaryFiles('../input/osic-pulmonary-fibrosis-progression/train/*/*.dcm').map(lambda x: getMap(x))
hasattr(rdd, "toDF")
                                                    
 # Converting RDD to Dataframe                                                    
df_image = rdd.toDF().cache()

def foldl(zero, combine, elements):
    if callable(zero):
        result = zero()
    else:
        result = zero
    for x in elements:
        result = combine(result, x)
    return result

def operator(df, elem):
    if elem == "Patient":
        df.withColumn(elem, df)
        
# Setting correct data Type for each column        
dfWithCols = df_image.select(df_image['_1.Patient'], df_image['_1.Weeks'], df_image['_1.AverageAspectRatio'],                df_image['_1.NumberOfContours'], df_image['_1.PercentageRatio'],                df_image['_1.TotalContourArea']).withColumn('Week', col('Weeks').cast(FloatType())).                withColumn('AverageAspectRatio', col('AverageAspectRatio').cast(FloatType())).                withColumn('NumberOfContours', col('NumberOfContours').cast(FloatType())).                withColumn('PercentageRatio', col('PercentageRatio').cast(FloatType())).                withColumn('TotalContourArea', col('TotalContourArea').cast(FloatType())).drop(col('Weeks'))

# Writing data to S3 bucket in parquet format, Parquest maintain the data types of each column while extracting again.
dfWithCols.write.mode('overwrite').partitionBy('Patient', 'Week').parquet('s3://osis-parquet/CTProps/props')


# In[28]:


dataframe = spark.read.parquet("../input/parquet/CTProps/final_data")
dataframe.toPandas()


# In[29]:


from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
from pyspark.ml.feature import Normalizer, VectorAssembler
import seaborn as sns
import pandas as pd
final_cols = [
    'Age',
    'Week',
    'TotalContourArea',
    'AverageAspectRatio',
    'PercentageRatio',
    'NumberOfContours',
    'FVC',
    'Percent'
]
fig, ax = plt.subplots(figsize=(10,10))
vector = VectorAssembler(
    inputCols= final_cols,
    outputCol= 'vector',
    handleInvalid="keep"
    ).transform(dataframe)
normalized_vector = Normalizer(inputCol='vector', outputCol='normalized_vector').transform(vector)
pearsonCorr = Correlation.corr(normalized_vector, 'normalized_vector').collect()[0][0].toArray()
df = pd.DataFrame(pearsonCorr, columns=final_cols )
sns.heatmap(df, xticklabels= final_cols, yticklabels= final_cols, ax = ax)
print("Following Observations are made from correlation matrix")
print("1. Total Contour Area is very negatively correlated with FVC")
print("2. Average Aspect Ratio is very positively correlated with FVC")
print("3. Percent Ratio is very negatively correlated with FVC")
print("4. Number Of Contours is very LESS correlated with FVC, Its value is around 0.0 - 0.25")
print("I think it is better to neglect Number Of contours from analysis")


# In[30]:



from pyspark.sql.functions import col
from pyspark.sql import functions as F
effective_dataframe = dataframe.join( dataframe.groupBy("Patient")                                    .agg(F.max("Week").alias("max_week")),                                    on=['Patient'],                                    how='inner').withColumn("effective_Week",                                                             col('Week')/col('max_week')*100)
effective_dataframe.toPandas()


# In[31]:



fig, axs = plt.subplots(3, 2, figsize=(20,20))
# pd_dataframe = dataframe.toPandas()
pd_dataframe = effective_dataframe.toPandas()
groups = pd_dataframe.groupby("SmokingStatus")

for name, group in groups:
    axs[0, 0].set_title('FVC vs Total Conntour Area', fontweight = 'bold' , fontsize = 14)
    axs[0, 0].plot(group["FVC"], group["TotalContourArea"], marker="o",alpha = 0.8, linestyle="", label=name)
axs[0, 0].legend()
axs[0, 0].set_xlabel("FVC", fontweight = 'bold' , fontsize = 14)
axs[0, 0].set_ylabel("Contour Area", fontweight = 'bold' , fontsize = 14)

for name, group in groups:
    axs[0, 1].set_title('FVC vs Percentage Ration of Area', fontweight = 'bold' , fontsize = 14)
    axs[0, 1].plot(group["FVC"], group["PercentageRatio"], marker="o",alpha = 0.8, linestyle="", label=name)
axs[0, 1].legend()
axs[0, 1].set_xlabel("FVC", fontweight = 'bold' , fontsize = 14)
axs[0, 1].set_ylabel("Percentage Ratio", fontweight = 'bold' , fontsize = 14)


for name, group in groups:
    axs[1, 0].set_title('Week vs Total Conntour Area', fontweight = 'bold' , fontsize = 14)
    axs[1, 0].plot(group["Week"], group["TotalContourArea"], marker="o",alpha = 0.8, linestyle="", label=name)
axs[1, 0].legend()
axs[1, 0].set_xlabel("Week", fontweight = 'bold' , fontsize = 14)
axs[1, 0].set_ylabel("Contour Area", fontweight = 'bold' , fontsize = 14)

for name, group in groups:
    axs[1, 1].set_title('Week vs Percentage Ration of Area', fontweight = 'bold' , fontsize = 14)
    axs[1, 1].plot(group["Week"], group["PercentageRatio"], marker="o",alpha = 0.8, linestyle="", label=name)
axs[1, 1].legend()
axs[1, 1].set_xlabel("Week", fontweight = 'bold' , fontsize = 14)
axs[1, 1].set_ylabel("Percentage Ratio", fontweight = 'bold' , fontsize = 14)



# for name, group in groups:
#     axs[1, 0].set_title('Week vs Total Conntour Area', fontweight = 'bold' , fontsize = 14)
#     axs[1, 0].plot(group["effective_Week"], group["TotalContourArea"], marker="o",alpha = 0.8, linestyle="", label=name)
# axs[1, 0].legend()
# axs[1, 0].set_xlabel("Week", fontweight = 'bold' , fontsize = 14)
# axs[1, 0].set_ylabel("Contour Area", fontweight = 'bold' , fontsize = 14)

# for name, group in groups:
#     axs[1, 1].set_title('Week vs Percentage Ration of Area', fontweight = 'bold' , fontsize = 14)
#     axs[1, 1].plot(group["effective_Week"], group["PercentageRatio"], marker="o",alpha = 0.8, linestyle="", label=name)
# axs[1, 1].legend()
# axs[1, 1].set_xlabel("Week", fontweight = 'bold' , fontsize = 14)
# axs[1, 1].set_ylabel("Percentage Ratio", fontweight = 'bold' , fontsize = 14)






for name, group in groups:
    axs[2, 0].set_title('Age vs Aspect Ratio', fontweight = 'bold' , fontsize = 14)
    axs[2, 0].plot(group["Age"], group["AverageAspectRatio"], marker="o",alpha = 0.8, linestyle="", label=name)
axs[2, 0].legend()
axs[2, 0].set_xlabel("Age", fontweight = 'bold' , fontsize = 14)
axs[2, 0].set_ylabel(" Aspect Ratio", fontweight = 'bold' , fontsize = 14)

for name, group in groups:
    axs[2, 1].set_title('FVC vs Aspect Ratio', fontweight = 'bold' , fontsize = 14)
    axs[2, 1].plot(group["FVC"], group["AverageAspectRatio"], marker="o",alpha = 0.8, linestyle="", label=name)
axs[2, 1].legend()
axs[2, 1].set_xlabel("FVC", fontweight = 'bold' , fontsize = 14)
axs[2, 1].set_ylabel("Aspect Ratio", fontweight = 'bold' , fontsize = 14)

plt.show()

print("Values of Percentage Ratio and Contour Area against FVC and Weeks are much lower for Patients who never smoked And is much higher for Ex smokers")

