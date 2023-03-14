#Linking drive to colab to store datasets
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse

# Generate auth tokens for Colab
from google.colab import auth
auth.authenticate_user()

# Generate creds for the Drive FUSE library. Though the link asks you to verify twice, you don't have to!
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

# Create a directory and mount Google Drive using that directory.
!mkdir -p drive
!google-drive-ocamlfuse drive

print 'Files in Drive:'
!ls drive/

# Create a file in a new directory called "Kaggle" in your Google Drive. This will be your operation base :P
!echo "This newly created file will appear in your Drive file list. If you are reading this, that means the attempt to integrate was successful" > drive/kaggle/created.txt

#The uploaded files are in .zip format. The following code will unzip them into nice json files. This has to be done only the first time!
!unzip "drive/kaggle/*.zip" -d drive/kaggle

#Now, remove those archives
!rm -f drive/kaggle/*.zip

#Make directories for the data
!mkdir drive/kaggle/train drive/kaggle/validation drive/kaggle/test

#Now, to download the train set into your drive from the urls in the JSON files, execute the below.  Also, a file is generated with the ImageURL, imageName, imageId and 
#their labelIds.


import json
import time

train_data = json.load(open('drive/kaggle/train.json'))
!echo ImageURL, ImgName, ImgId, LabelId >> drive/kaggle/train/train.txt

for i in range(len(train_data['images'])):
  img_url = train_data['images'][i]['url']
  img_id = train_data['images'][i]['imageId']
  label_id = train_data['annotations'][i]['labelId']
  img_name=img_url.split("/")[-1]
  #print img_name
  img_name_actual = img_name.split("-")[0]
  img_name_small = img_name_actual + "-small"+".jpg"
  #print img_name_actual
  img_url_small = img_url.split("-")[-2]
  img_url_small = img_url_small + "-small"
  print img_url_small
  !curl $img_url_small > drive/kaggle/train/$img_name_small
  time.sleep(0.05) 
  !echo $img_url_small,$img_name_small,$img_id,$label_id >> drive/kaggle/train/train.txt 
  #time.sleep(0.5)

#To download validation data on to your drive...

import json
import time

val_data = json.load(open('drive/kaggle/validation.json'))
!echo ImageURL, ImgName, ImgId, LabelId >> drive/kaggle/validation/validation.txt 

for i in range(len(val_data['images'])):
  img_url = val_data['images'][i]['url']
  #print img_url
  img_id = val_data['images'][i]['imageId']
  #print img_id
  label_id = val_data['annotations'][i]['labelId']
  #print label_id
  img_name=img_url.split("/")[-1]
  #print img_name
  img_name_actual = img_name.split("-")[0]
  img_name_small = img_name_actual + "-small"+".jpg"
  #print img_name_actual
  img_url_small = img_url.split("-")[-2]
  img_url_small = img_url_small + "-small"
  print img_url_small
  !curl $img_url_small > drive/kaggle/validation/$img_name_small
  time.sleep(0.05)
  !echo $img_name_actual,$img_id,$label_id >> drive/kaggle/validation/validation.txt 
  #time.sleep(0.05)

#And this is for downloading test data into your drive

import json
import time

test_data = json.load(open('drive/kaggle/test.json'))
#print len(test_data['images'])

for i in range(len(test_data['images'])):
  img_url = test_data['images'][i]['url']
  #print img_url
  img_id = test_data['images'][i]['imageId']
  #print img_id
  img_name=img_url.split("/")[-1]
  #print img_name
  img_name_actual = img_name.split("-")[0]
  img_name_small = img_name_actual + "-small"+".jpg"
  #print img_name_actual
  img_url_small = img_url.split("-")[-2]
  img_url_small = img_url_small + "-small"
  print img_url_small
  !curl $img_url_small > drive/kaggle/test/$img_name_small
  time.sleep(0.05)
  
