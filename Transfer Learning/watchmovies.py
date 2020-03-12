# -*- coding: utf-8 -*-
"""
@author: Giba1
"""
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

window=16 # Number of frames in sliding window.

import sys
from skimage import io, transform
import matplotlib.animation as animation

train = pd.read_csv('driver_imgs_list.csv')
train['id'] = range(train.shape[0])
fig = plt.figure()
subj = np.unique(train['subject'])[0]

class frame:
    def __init__(self, subject,pixels,classlabel):
        self.pixels = pixels
        self.subject = subject
        self.classlabel = classlabel

clips=[]
cliplabels=[]
pixels=[]
labels=[]
subject=[]

frames=[]

i=0
numsubjects=0

oldexpression='e'
oldindex=0
cliplengths=[]

for subj in np.unique(train['subject'])[20:]:

    imagem = train[train['subject'] == subj]

    imgs = []
    t = imagem.values[0]
    for t in imagem.values:
        if (oldexpression!= [int(t[1][1:])]):

            if (oldexpression!='e'):
                print 'segment', i, subj, oldexpression, ' size: ',(i-oldindex)
                cliplengths.append(i-oldindex)
                print np.max(cliplengths), np.min(cliplengths), np.mean(cliplengths), len(cliplengths),len(clips)

            oldexpression=[int(t[1][1:])]
            oldindex=i
            if (i !=0):
                clips.append(np.asarray(frames))
                cliplabels.append(int(t[1][1:]))
            frames=[]
            img = cv2.imread('imgs/train/' + t[1] + '/' + t[2], 3)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.resize(gray_image,(112,112))
            frames.append(np.asarray(gray_image))
            print np.shape(np.asarray(gray_image))
        else:
            img = cv2.imread('imgs/train/' + t[1] + '/' + t[2], 3)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.resize(gray_image,(112,112),interpolation=cv2.INTER_AREA)

            frames.append(np.asarray(gray_image))

        i=i+1
    numsubjects=numsubjects+1

print len(cliplabels)
print len(clips)

Y=[]
X=[]
j=0

for clip in clips: # expand clips with sliding window.
    for i in xrange(0,len(clip)-window, 8):
        X.append(np.asarray(clip[i:window+i]))
        Y.append(cliplabels[j])
    j=j+1




X=np.asarray(X, dtype='uint8')
Y=np.asarray(Y,dtype='uint8')

print np.shape(X)

#
# dups=0
#
# for jjj in xrange(0,len(X)):
#     XP=X[jjj]
#     for iii in xrange(0,len(X)):
#         if(np.all(XP==X[iii])):
#             dups=dups+1
#     print dups,len(X)

#print len(pixels)
#print len(labels)

print i

p=np.unique(X)

print 'what is going on?'

np.save('valclipssmall.np',X)
np.save('vallabelssmall.np',Y)

#np.save('valclipssmall.np',X)
#np.save('vallabelssmall.np',Y)

    # ax = fig.add_subplot(111)
    # ax.set_axis_off()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)  # removes white border
    # fname = 'MOVIE_subject_' + subj + '.gif'
    # imgs = [(ax.imshow(img),
    #          ax.set_title(t[0]),
    #          ax.annotate(n_img, (5, 5))) for n_img, img in enumerate(imgs)]
    # img_anim = animation.ArtistAnimation(fig, imgs, interval=125,
    #                                      repeat_delay=1000, blit=False)
    # print('Writing:', fname)
    # img_anim.save(fname, writer='imagemagick', dpi=20)
    # fig.clf()
print ('Now relax and watch some movies!!!')


#data[1].classlabel
#print data[1].subject