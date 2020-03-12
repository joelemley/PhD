import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

#caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(1)


#net = caffe.Net('train_test2.prototxt', caffe.TRAIN)

solver = caffe.get_solver("solver2.prototxt")
solver.solve()
