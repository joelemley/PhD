from scipy import io
import numpy
import tables
import os
import cv2
import h5py
import sys
import numpy as np

import random

def create_h5_file(labels,images,file_name):
    with h5py.File(file_name, "w") as H:
        H.create_dataset("data", data=images)
        H.create_dataset("label", data=labels)

'''Rodrigues formula
Input: 1x3 array of rotations about x, y, and z
Output: 3x3 rotation matrix'''
from numpy import array,mat,sin,cos,dot,eye
from numpy.linalg import norm
from os import listdir
from PIL import Image

import lmdb
import caffe

def rodrigues(r):
    def S(n):
        Sn = array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
        return Sn
    theta = norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = S(n)
        R = eye(3) + sin(theta)*Sn + (1-cos(theta))*dot(Sn,Sn)
    else:
        Sr = S(r)
        theta2 = theta**2
        R = eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*dot(Sr,Sr)
    return mat(R)


def changeResolution(left,right,fromresolution,toresolution):

    # Input image 60 x 36 -> Downscale to 52 x 31 -> Upscale to 60 x 36 -> CNN Eye gaze angle
    # Input image 60 x 36 -> Downscale to 26 x 16 -> Upscale to 60 x 36 -> CNN Eye gaze angle.
    left = Image.fromarray(np.asarray(left))
    left = left.resize(toresolution)
    left = left.resize(fromresolution)
    right = Image.fromarray(np.asarray(right))
    right = right.resize(toresolution)
    right = right.resize(fromresolution)

    return np.asarray(left),np.asarray(right)

def makelbl(matfile,dataindex):
    gazeright=matfile["data"]["right"][0][0][0][0]["gaze"][dataindex]
    poseright=matfile["data"]["right"][0][0][0][0]["pose"][dataindex]
    poseleft=matfile["data"]["left"][0][0][0][0]["pose"][dataindex]
    gazeleft=matfile["data"]["left"][0][0][0][0]["gaze"][dataindex]
    gaze=gazeleft
    pose=poseleft
    for tt in xrange(0,len(gazeleft)):
        gaze[tt]=(gaze[tt]+gazeright[tt])/2
        pose[tt] = (pose[tt] + poseright[tt]) / 2
    theta = np.arcsin(-1 * gaze[1])
    phi = np.arctan2(-1 * gaze[0], -1 * gaze[2])
    gaze = np.asarray([theta, phi])
    #
    M = rodrigues((pose[0], pose[1], pose[2]))
    Zv = M[:, 2]
    theta2 = np.arcsin(np.squeeze(Zv[1]))
    phi2 = np.arctan2(np.squeeze(Zv[0]), np.squeeze(Zv[2]))
    theta2 = theta2.A1
    phi2 = phi2.A1
    pose = np.concatenate((theta2, phi2))
    lbl = np.concatenate((gaze, pose))
    lbl = lbl / 1.0
    return lbl.astype(np.float32)


def getData(matfile,dataindex,augRes=None):

    leftimg=matfile["data"]["left"][0][0][0][0]["image"][dataindex]
    rightimg = matfile["data"]["right"][0][0][0][0]["image"][dataindex]
    if augRes is not None:
        leftimg,rightimg=changeResolution(leftimg,rightimg,np.shape(leftimg),augRes)


    leftimg= cv2.equalizeHist(np.asarray(leftimg))
    rightimg= cv2.equalizeHist(np.asarray(rightimg))

    image=[leftimg,rightimg]
    image=np.asarray(image)/255.0

    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg
    # #
    # fig = plt.figure(figsize=(5, 5))
    # sub = fig.add_subplot(2, 1, 0 + 1)
    # sub.imshow(np.asarray(leftimg), cmap="gray")
    # sub = fig.add_subplot(2, 1, 1 + 1)
    # sub.imshow(np.asarray(rightimg), cmap="gray")
    # #
    # fig.show()
    # plt.show()

#
    lbl=makelbl(matfile,dataindex)
    image=image.astype(np.float32)

    return lbl,np.reshape(image, (2, 36, 60))

def getAugmentedData(matfile,dataindex):
    resolutions=[]

#    for i in xrange(35,13,-1):
#        y = round((i / 0.6), 0)
#        resolutions.append((i,int(y)))

    x=random.randint(14,35)
    y=int(round(x/0.6,0))
    resolutions.append((x,y))

    x=random.randint(14,35)
    y=int(round(x/0.6,0))
    resolutions.append((x,y))

    labels = []
    images = []
    for res in resolutions:
        lbl, image = getData(matfile, dataindex,augRes=res)
        labels.append(lbl)
        images.append(image)
    return labels,images



def prepareData(person):
    data = ""
    samples = []

    o=listdir("MPIIGaze/Data/Normalized/p"+person+"/")

    labels = []
    images = []
    for mfile in o:

        matlocation="MPIIGaze/Data/Normalized/p"+person+"/"+mfile
        matfile=io.loadmat(matlocation)
        numimages=len(matfile["data"]["right"][0][0][0][0]["image"])
        for dataindex in xrange(0,numimages):
            lbl, image=getData(matfile,dataindex)
            labels.append(lbl)
            images.append(image)
            l,im=getAugmentedData(matfile,dataindex)
            labels.extend(l)
            images.extend(im)
    return labels,images

def makeLmDBfile(X,Y):
    map_size = X.nbytes * 10

    env = lmdb.open('mylmdb', map_size=map_size)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(len(Y)):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = Y[i]
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

X=[]
Y=[]
print "Augmenting with the following resolutions:"
for i in xrange(35,13,-1):

    y=round((i / 0.6), 0)
    print((i,int(y)))
for i in xrange(0,3):
    ind=str(i)
    if (len(ind)<2):
        ind="0"+ind
    print "processing "+ind
    labels,images=prepareData(ind)
    print "concatenating "+ind+".m5"
    if i==0:
        X=images
        Y=labels
        print np.shape(Y)
    else:
        X=np.concatenate((X,images),axis=0)
        Y=np.concatenate((Y, labels),axis=0)
        print np.shape(Y)

print len(X)
print len(Y)
print np.shape(Y)
makeLmDBfile(X,Y)


#create_h5_file(labels,images,ind+"augdc.m5")

#io.savemat("p00",)
print "hello"
