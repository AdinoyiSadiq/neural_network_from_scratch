from zipfile import ZipFile
import os
import urllib
import urllib.request

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = '../data/fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

# Ensure the directory exists
if not os.path.exists(os.path.dirname(FILE)):
    os.makedirs(os.path.dirname(FILE))

# Ensure the file exists
if not os.path.isfile(FILE):
  print(f'Downloading {URL} and saving as {FILE}...') 
  urllib.request.urlretrieve(URL, FILE)

print('Unzipping images...')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

print('Done!')