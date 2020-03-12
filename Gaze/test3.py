from scipy import io
import numpy
import tables
import os

import h5py
import sys
import numpy as np

def create_h5_file(labels,images,file_name):
    with h5py.File(file_name, "w") as H:
        H.create_dataset("data", data=images)
        H.create_dataset("label", data=labels)



'''Rodrigues formula
Input: 1x3 array of rotations about x, y, and z
Output: 3x3 rotation matrix'''
from numpy import array,mat,sin,cos,dot,eye
from numpy.linalg import norm

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

data=""

person="00"

samples=[]

with open ("MPIIGaze/Evaluation Subset/p"+person+".txt", "r") as myfile:
    data = myfile.readlines()

labels=[]
images=[]

for sampleid in xrange(0,1):

    day= data[sampleid][0:5]
    whicheye=data[sampleid][15:].strip()
    dataindex=int(data[sampleid][6:10])-1# the -1 is to correct for matlab indexing. 0001 becomes 0000

    matlocation="MPIIGaze/Data/Normalized/p"+person+"/"+day+".mat"
    matfile=io.loadmat(matlocation)

    image=matfile["data"][whicheye][0][0][0][0]["image"][dataindex]
    image=image/255.0
    gaze=matfile["data"][whicheye][0][0][0][0]["gaze"][dataindex]
    pose=matfile["data"][whicheye][0][0][0][0]["pose"][dataindex]

    theta = np.arcsin(-1*gaze[1])
    phi = np.arctan2(-1*gaze[0], gaze[2])
    gaze=np.asarray([theta,phi])
#    print gaze
    M = rodrigues((pose[0],pose[1],pose[2]))
    Zv = M[:,2]
    theta2 = np.arcsin(np.squeeze(Zv[1]))
    phi2 = np.arctan2(np.squeeze(Zv[0]), np.squeeze(Zv[2]))
    theta2=theta2.A1
    phi2=phi2.A1


    pose=np.concatenate((theta2,phi2))
    print theta2, phi2
#    print M
#    print pose
    lbl=np.concatenate((gaze,pose))
    lbl=lbl/1.0
    lbl=lbl.astype(np.float32)
    image=image.astype(np.float32)
    labels.append(lbl)
    images.append(np.reshape(image,(1,36,60)))

create_h5_file(labels,images,"mpiigaze.m5")

#io.savemat("p00",)
print "hello"
