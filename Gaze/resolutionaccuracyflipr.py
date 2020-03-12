from scipy import io
import numpy
import tables
import os
import cv2
import h5py
import sys
import numpy as np
from PIL import Image

def create_h5_file(labels,images,file_name):
    with h5py.File(file_name, "w") as H:
        H.create_dataset("data", data=images)
        H.create_dataset("label", data=labels)


#Input image 60 x 36 -> Downscale to 52 x 31 -> Upscale to 60 x 36 -> CNN Eye gaze angle
#Input image 60 x 36 -> Downscale to 26 x 16 -> Upscale to 60 x 36 -> CNN Eye gaze angle.

'''Rodrigues formula
Input: 1x3 array of rotations about x, y, and z
Output: 3x3 rotation matrix'''
from numpy import array,mat,sin,cos,dot,eye
from numpy.linalg import norm
from os import listdir
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

    left = Image.fromarray(left)
    left = left.resize(toresolution,resample=Image.LANCZOS)
    left = left.resize(fromresolution,resample=Image.LANCZOS)
    right = Image.fromarray(right)
    right = right.resize(toresolution,resample=Image.LANCZOS)
    right = right.resize(fromresolution,resample=Image.LANCZOS)

    return np.asarray(left),np.asarray(right)

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

            leftimg=matfile["data"]["left"][0][0][0][0]["image"][dataindex]
            rightimg = matfile["data"]["right"][0][0][0][0]["image"][dataindex]
            rightimg=np.flip(rightimg,1)
            # Input image 60 x 36 -> Downscale to 52 x 31 -> Upscale to 60 x 36 -> CNN Eye gaze angle
            # Input image 60 x 36 -> Downscale to 26 x 16 -> Upscale to 60 x 36 -> CNN Eye gaze angle.
            a=np.shape(leftimg)
            leftimg,rightimg=changeResolution(leftimg,rightimg,np.shape(leftimg),(31,52))


            gazeright=matfile["data"]["right"][0][0][0][0]["gaze"][dataindex]
            poseright=matfile["data"]["right"][0][0][0][0]["pose"][dataindex]

            poseleft=matfile["data"]["left"][0][0][0][0]["pose"][dataindex]
            gazeleft=matfile["data"]["left"][0][0][0][0]["gaze"][dataindex]

            gaze=gazeleft
            pose=poseleft
            for tt in xrange(0,len(gazeleft)):
                gaze[tt]=(gaze[tt]+gazeright[tt])/2
                pose[tt] = (pose[tt] + poseright[tt]) / 2
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
            theta = np.arcsin(-1 * gaze[1])
            phi = np.arctan2(-1 * gaze[0], -1* gaze[2])
            gaze = np.asarray([theta, phi])
    #
            M = rodrigues((pose[0], pose[1], pose[2]))
            Zv = M[:, 2]
            theta2 = np.arcsin(np.squeeze(Zv[1]))
            phi2 = np.arctan2(np.squeeze(Zv[0]), np.squeeze(Zv[2]))
            theta2 = theta2.A1
            phi2 = phi2.A1
            pose = np.concatenate((theta2, phi2))
            lbl=np.concatenate((gaze,pose))
            lbl=lbl/1.0
            lbl=lbl.astype(np.float32)
            image=image.astype(np.float32)
            labels.append(lbl)
            images.append(np.reshape(image,(2,36,60)))
    return labels,images



ind="06"
labels,images=prepareData(ind)
print "creating "+ind+".m5"
create_h5_file(labels,images,"resolutiontestf1.m5")

#io.savemat("p00",)
print "hello"
