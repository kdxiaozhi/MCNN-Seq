import os
import numpy as np
from osgeo import gdal
import pickle
from configs import args
def read_img(img_path):
    dataset = gdal.Open(img_path)
    samples = dataset.RasterXSize
    lines = dataset.RasterYSize
    bands = dataset.RasterCount
    geotans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    data = np.zeros([bands, lines, samples, ])
    for i in range(bands):
        band_data = dataset.GetRasterBand(i + 1)
        data[i, :, :] = band_data.ReadAsArray(0, 0, samples, lines)
    return data,lines,samples,bands,geotans,proj

def image_to_list(image):
    row = image.shape[1]
    col = image.shape[2]
    band = image.shape[0]
    image_pixel_all = []
    for i in range(row):
        for j in range(col):
            image_pixel= image[:,i,j]
            image_pixel_all.append(image_pixel)
    image_list = np.stack(image_pixel_all, axis=0)
    return image_list
#read img
dataset_img,lines,samples,bands,geotans,proj=read_img(args.img_path)

dataset_list=image_to_list(dataset_img)

#tr&te
cloud_label=dataset_list[:,0]
img=dataset_list[:,1:]
label_unique=np.unique(cloud_label)
print(label_unique)
tr_id=np.where(cloud_label == label_unique[1])
tr=np.squeeze(img[tr_id,:])

te_id=np.where(cloud_label == label_unique[0])
te=np.squeeze(img[te_id,:])


#x,y

S2_tr=tr[:,:args.lenght]
S1_tr=tr[:,args.lenght:]
S2_te=te[:,:args.lenght]
S1_te=te[:,args.lenght:]

for i in ["S1_tr","S2_tr","S1_te","S2_te"]:
    x_pickle = open( i+'.pickle', 'wb')
    pickle.dump(eval(i), x_pickle)
    x_pickle.close()
    
    
    
    