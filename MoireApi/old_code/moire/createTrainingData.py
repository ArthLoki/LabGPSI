import sys
import argparse
from PIL import Image
from PIL import ImageOps
import random
import sys
import os

from os import listdir
from os.path import isfile, join
from PIL import Image
from moire.haar2D import fwdHaarDWT2D


#The wavelet decomposed images are the transformed images representing the spatial and the frequency information of the image. These images are stored as 'tiff' in the disk, to preserve that information. Each image is transformed with 180 degrees rotation and as well flipped, as part of data augmentation.
def transformImageAndSave(image, f, customStr, path):
    cA, cH, cV, cD  = fwdHaarDWT2D(image);
    
    fileName = (os.path.splitext(f)[0])
    fLL = (f.replace(fileName, fileName+'_' + customStr + 'LL')).replace('.jpg','.tiff')
    fLH = (f.replace(fileName, fileName+'_' + customStr + 'LH')).replace('.jpg','.tiff')
    fHL = (f.replace(fileName, fileName+'_' + customStr + 'HL')).replace('.jpg','.tiff')
    fHH = (f.replace(fileName, fileName+'_' + customStr + 'HH')).replace('.jpg','.tiff')
    cA = Image.fromarray(cA)
    cH = Image.fromarray(cH)
    cV = Image.fromarray(cV)
    cD = Image.fromarray(cD)
    cA.save(join(path, fLL))
    cH.save(join(path, fLH))
    cV.save(join(path, fHL))
    cD.save(join(path, fHH))

def augmentAndTrasformImage(f, mainFolder, trainFolder):
    try:
        img = Image.open(join(mainFolder, f)) 
    except Exception as e:
        print('Error: Couldnt read the file {}. Make sure only images are present in the folder'.format(f))
        return None
        
    w, h = img.size
    if h > w:
        img = img.resize((750, 1000))
    else:
        img = img.resize((1000, 750))

    imgGray = img.convert('L')
    wdChk, htChk = imgGray.size
    if htChk > wdChk:
        imgGray = imgGray.rotate(-90, expand=1)
        print('training image rotated')
    transformImageAndSave(imgGray, f, '', trainFolder)

    imgGray = imgGray.transpose(Image.ROTATE_180)
    transformImageAndSave(imgGray, f, '180_', trainFolder)

    imgGray = imgGray.transpose(Image.FLIP_LEFT_RIGHT)
    transformImageAndSave(imgGray, f, '180_FLIP_', trainFolder)
    
    return True
