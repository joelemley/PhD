#22nd to 3rd

import numpy as np
import scipy.io
import time
import pickle
import lasagne
import theano.tensor as T
import theano
import sys
import random

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax

random.seed(100)
sys.setrecursionlimit(10000)

#plotting

import matplotlib.pyplot as plt
import numpy as np
plt.ion() ## Note this correction
fig=plt.figure()

i=0
x=list()
y=list()

#Database Parameters
train1=np.fromfile('trainclass1',dtype='uint8')

train1=np.reshape(train1,(15000,3,256,256))
train1=np.reshape(train1,(15000,256,256,3))
train1=np.transpose((train1),(0,3,1,2))

train2=np.fromfile('trainclass2',dtype='uint8')
train2=np.reshape(train2,(15000,3,256,256))
train2=np.reshape(train2,(15000,256,256,3))
train2=np.transpose((train2),(0,3,1,2))

validate1 = np.fromfile('valclass1',dtype='uint8')
validate1 = np.reshape(validate1,(100,3,256,256))
validate1 = np.reshape(validate1,(100,256,256,3))
validate1 = np.transpose((validate1),(0,3,1,2))

validate2 = np.fromfile('valclass2',dtype='uint8')
validate2 = np.reshape(validate2,(100,3,256,256))
validate2 = np.reshape(validate2,(100,256,256,3))
validate2 = np.transpose((validate2),(0,3,1,2))

imageW=256
imageH=256

TrainData = {'imClass1':train1,'imClass2':train2}
ValidationData = {'imClass1':validate1,'imClass2':validate2}

#Training Parameters

num_epochs = 1000
validationErrorBest = 100000;
validationAccuracyBest = 0;
alpha = .3
beta = .7
netAnumInputs = 2
batchSize = 5

numSamplesTrain=len(TrainData["imClass1"])+len(TrainData["imClass2"])
numSamplesValidation=len(ValidationData["imClass1"])+len(ValidationData["imClass2"])

class BatchNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, axes=None, epsilon=0.01, alpha=0.5,
                 nonlinearity=None, **kwargs):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).

        @param incoming: `Layer` instance or expected input shape
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        dtype = theano.config.floatX
        self.mean = self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(lasagne.init.Constant(1), shape, 'std',
                                  trainable=False, regularizable=False)
        self.beta = self.add_param(lasagne.init.Constant(0), shape, 'beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(lasagne.init.Constant(1), shape, 'gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = input.mean(self.axes, keepdims=True)
            std = input.std(self.axes, keepdims=True)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * std)
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            std += 0 * running_std
        std += self.epsilon
        mean = T.addbroadcast(mean, *self.axes)
        std = T.addbroadcast(std, *self.axes)
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        normalized = (input - mean) * (gamma / std) + beta
        return self.nonlinearity(normalized)


def batch_norm(layer , **kwargs):
    """
    Convenience function to apply batch normalization to a given layer's output.
    Will steal the layer's nonlinearity if there is one (effectively introducing
    the normalization right before the nonlinearity), and will remove the
    layer's bias if there is one (because it would be redundant).
    @param layer: The `Layer` instance to apply the normalization to; note that
        it will be irreversibly modified as specified above
    @return: A `BatchNormLayer` instance stacked on the given `layer`
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity,**kwargs)

'''
def Give_Right_Data_MultiNet(TrainValData , batchSize , NetAinputNums,SA):

    DataX = np.empty(shape=[batchSize, NetAinputNums, imageW, imageH])
    DataY = np.empty(shape=[batchSize, 1, imageW, imageH])
    classPerm = np.empty([batchSize, 1])
    for ii in range(batchSize):
        classPerm[ii, 0] = np.int(np.random.random() * 2)

    for ii in range(batchSize):
        classNum = int(classPerm[ii, 0])
        className = 'imClass' + str(classNum + 1)
        samplePerm = np.random.permutation(TrainValData[className].shape[0])
        tempBatchX = TrainValData[className][samplePerm[0:NetAinputNums], :, :, :]
        tempBatchY = TrainValData[className][samplePerm[NetAinputNums:NetAinputNums + 1], :, :, :]
        for jj in range(NetAinputNums):
            DataX[ii: (ii + 1), jj: (jj + 1), :, :] = tempBatchX[jj, :, :, :]
        DataY[ii: ii + 1, :, :, :] = tempBatchY

    Class1Flags = np.zeros(shape=[batchSize, NetAinputNums, imageW, imageH])
    Class2Flags = np.zeros(shape=[batchSize, NetAinputNums, imageW, imageH])
    for ii in range(batchSize):
        if classPerm[ii] == 0:
            Class1Flags[ii,:,:,:]=1
        if classPerm[ii] == 1:
            Class2Flags[ii,:,:,:]=1
    if(SA==1):
        return np.float32(DataX), np.float32(DataY), np.uint8(classPerm), np.float32(Class1Flags) , np.float32(Class2Flags)
    else:
        return np.float32(DataX), np.float32(DataY), np.uint8(np.concatenate((classPerm, classPerm))) , np.float32(Class1Flags) , np.float32(Class2Flags)


'''

def Give_Right_Data_SA_2(TrainValData , batchSize , NetAinputNums):#classPerm = np.random.permutation(10)
#batchSize = 100
#NetAinputNums = 5
    DataX = np.empty(shape=[batchSize,  NetAinputNums*3, imageW, imageH])
    DataY = np.empty(shape=[batchSize, 3, imageW, imageH])
    classPerm = np.empty([batchSize, 1])
    for ii in range(batchSize):
        classPerm[ii, 0] = np.int(np.random.random() * 2)

    for ii in range(batchSize):
        classNum = int(classPerm[ii, 0])
        className = 'imClass' + str(classNum + 1)
        samplePerm = np.random.permutation(TrainValData[className].shape[0])
        tempBatchX = TrainValData[className][samplePerm[0:NetAinputNums], :, :, :]
        tempBatchY = TrainValData[className][samplePerm[NetAinputNums:NetAinputNums + 1], :, :, :]
        for jj in range(NetAinputNums):
            DataX[ii: (ii + 1), jj*3 : (jj + 1)*3 , :, :] = tempBatchX[jj, :, :, :]
        DataY[ii: ii + 1, :, :, :] = tempBatchY
    return np.float32(DataX)/np.float32(255) , np.float32(DataY)/np.float32(255) , np.uint8(np.concatenate((classPerm,classPerm)))

def Give_Right_Data_SA(TrainValData , batchSize , NetAinputNums):#classPerm = np.random.permutation(10)
#batchSize = 100
#NetAinputNums = 5
    DataX = np.empty(shape=[batchSize,  NetAinputNums*3, imageW, imageH])
    DataY = np.empty(shape=[batchSize, 3, imageW, imageH])
    classPerm = np.empty([batchSize, 1])
    for ii in range(batchSize):
        classPerm[ii, 0] = np.int(np.random.random() * 2)

    for ii in range(batchSize):
        classNum = int(classPerm[ii, 0])
        className = 'imClass' + str(classNum + 1)
        samplePerm = np.random.permutation(TrainValData[className].shape[0])
        tempBatchX = TrainValData[className][samplePerm[0:NetAinputNums], :, :, :]
        tempBatchY = TrainValData[className][samplePerm[NetAinputNums:NetAinputNums + 1], :, :, :]
        for jj in range(NetAinputNums):
            DataX[ii: (ii + 1), jj*3 : (jj + 1)*3 , :, :] = tempBatchX[jj, :, :, :]
        DataY[ii: ii + 1, :, :, :] = tempBatchY
    return np.float32(DataX)/np.float32(255) , np.float32(DataY)/np.float32(255) , np.uint8(classPerm)


print('Building the model...')
input_var = T.tensor4('inputs')
input_varB = T.tensor4('inputsB')
target_varA = T.tensor4('targetsA')
target_varB = T.ivector('targetsB')
'''
networkAi = lasagne.layers.InputLayer(shape=(None , netAnumInputs , None , None) , name='networkAi')
netAL1 = lasagne.layers.Conv2DLayer(networkAi,num_filters=16,filter_size=(3,3),pad='same',name='netAL1')
#netL1BN = batch_norm(netL1 , name='netL1BN')
netAL2 = lasagne.layers.Conv2DLayer(netAL1 , num_filters=16,filter_size=(3,3),pad='same',name='netAL2')
#netL2BN = batch_norm(netL2 , name='netL2BN')
netAL3 = lasagne.layers.Conv2DLayer(netAL2,num_filters=32,filter_size=(3,3),pad='same',name='netAL3')
#netL3BN = batch_norm(netL3 , name='netL3BN')
netAL4 = lasagne.layers.Conv2DLayer(netAL3 , num_filters=32,filter_size=(3,3),pad='same',name='netAL4')
#netL4BN = batch_norm(netL4 , name='netL4BN')
networkAOut = lasagne.layers.Conv2DLayer(netAL4,num_filters=1,filter_size=(3,3),pad='same',name='networkAOut')
'''
networkBi = lasagne.layers.InputLayer(shape=(None,3,imageW,imageH),name='networkBi')
netBL1 = lasagne.layers.Conv2DLayer(networkBi,num_filters=16,filter_size=(3,3),name='netBL1')
netBL1BN = batch_norm(netBL1 , name='netBL1BN')
netBL1MP = lasagne.layers.MaxPool2DLayer(netBL1BN , pool_size=(3,3),name='netBL1MP')
netBL2 = lasagne.layers.Conv2DLayer(netBL1 , num_filters=8,filter_size=(3,3),name='netBL2')
netBL2BN = batch_norm(netBL2 , name='netBL2BN')
netBL2MP = lasagne.layers.MaxPool2DLayer(netBL2BN , pool_size=(3,3),name='netBL2MP')
netBL3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL2MP),num_units=1024,name='netBL3')
networkBOut = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL3),num_units=2,nonlinearity=lasagne.nonlinearities.softmax,name='networkBOut')

net = {}
net['input'] = InputLayer((None, 3, imageW, imageH))
net['conv1_1'] = ConvLayer(
    net['input'], 64, 3, pad=1, flip_filters=False)
net['conv1_2'] = ConvLayer(
    net['conv1_1'], 64, 3, pad=1, flip_filters=False)
net['pool1'] = PoolLayer(net['conv1_2'], 2)
net['conv2_1'] = ConvLayer(
    net['pool1'], 128, 3, pad=1, flip_filters=False)
net['conv2_2'] = ConvLayer(
    net['conv2_1'], 128, 3, pad=1, flip_filters=False)
net['pool2'] = PoolLayer(net['conv2_2'], 2)
net['conv3_1'] = ConvLayer(
    net['pool2'], 256, 3, pad=1, flip_filters=False)
net['conv3_2'] = ConvLayer(
    net['conv3_1'], 256, 3, pad=1, flip_filters=False)
net['conv3_3'] = ConvLayer(
    net['conv3_2'], 256, 3, pad=1, flip_filters=False)
net['pool3'] = PoolLayer(net['conv3_3'], 2)
net['conv4_1'] = ConvLayer(
    net['pool3'], 512, 3, pad=1, flip_filters=False)
net['conv4_2'] = ConvLayer(
    net['conv4_1'], 512, 3, pad=1, flip_filters=False)
net['conv4_3'] = ConvLayer(
    net['conv4_2'], 512, 3, pad=1, flip_filters=False)
net['pool4'] = PoolLayer(net['conv4_3'], 2)
net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
net['pool5'] = PoolLayer(net['conv5_3'], 2)
net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=2, nonlinearity=None)
net['prob'] = NonlinearityLayer(net['fc8'], softmax)

networkBOut=net['prob']
#outputA = lasagne.layers.get_output(networkAOut , inputs=input_var)
#loss = lasagne.objectives.squared_error(output , target_var)
#networkBin = T.concatenate((outputA,target_varA))
#lossA = lasagne.objectives.squared_error(outputA , target_varA)

#lossA = lossA.mean()

outputB = T.add(lasagne.layers.get_output(networkBOut , inputs=input_varB),np.finfo(np.float32).eps)
lossB = lasagne.objectives.categorical_crossentropy(outputB,target_varB)
lossB = lossB.mean()
loss_total = lossB
#paramsA = lasagne.layers.get_all_params(networkAOut , trainable = True)
#updatesA = lasagne.updates.nesterov_momentum(lossA , paramsA , learning_rate = 0.01 , momentum=0.9)

paramsB = lasagne.layers.get_all_params(networkBOut , trainable = True)
#updatesB = lasagne.updates.nesterov_momentum(lossB , paramsB , learning_rate = 0.01 , momentum=0.9)
#updates = lasagne.updates.sgd(loss , params , learning_rate = 0.01)
params_total  = paramsB#list(np.concatenate((paramsA , paramsB)))

updates = lasagne.updates.nesterov_momentum(loss_total , params_total , learning_rate=.0005,momentum=0.5)
#test_outputA = lasagne.layers.get_output(networkAOut ,inputs=input_var, deterministic = True)
#test_loss = lasagne.objectives.squared_error(test_output , target_var)
#test_lossA = lasagne.objectives.squared_error(test_outputA , target_varA)
#test_lossA = test_lossA.mean()

test_outputB = T.add(lasagne.layers.get_output(networkBOut ,inputs=input_varB, deterministic = True),np.finfo(np.float32).eps)
#test_loss = lasagne.objectives.squared_error(test_output , target_var)
test_lossB = lasagne.objectives.categorical_crossentropy(test_outputB , target_varB)
test_lossB = test_lossB.mean()
test_acc = T.mean(T.eq(T.argmax(test_outputB, axis=1), target_varB),
                  dtype=theano.config.floatX)

#loss_total = alpha*lossA+beta*lossB
train_func = theano.function([input_varB , target_varB] , loss_total , updates=updates)
valid_func = theano.function([input_varB , target_varB] , [test_lossB , test_acc])

print('Training...')
TRAINLOSS = np.array([])
VALIDATIONLOSS = np.array([])
VALODATIONACC = np.array([])
TESTACC = np.array([])
VALIDATIONACC = np.array([])
num_epochs = 1000
validationErrorBest = 100000;
for epoch in range(num_epochs):

    train_err = 0
    train_batches = 0

    start_time = time.time()
    for kk in range(int(numSamplesTrain/batchSize)):
        inputs , targetsA , targetsB = Give_Right_Data_SA(TrainData,batchSize=batchSize,NetAinputNums=netAnumInputs)
        targetsB = np.squeeze(targetsB)
        train_err += train_func(targetsA , targetsB)
        train_batches += 1
        #err = valid_func(inputs , targets)
    val_acc = 0
    val_err = 0
    val_batches = 0

    for pp in range(int(numSamplesValidation/batchSize)):
        inputs , targetsA , targetsB = Give_Right_Data_SA(ValidationData,batchSize=batchSize*2,NetAinputNums=netAnumInputs)
        targetsB = np.squeeze(targetsB)
        err , acc  = valid_func(targetsA , targetsB)
        val_err += err
        val_acc += acc
        val_batches +=1

    validationError = val_err / val_batches
    if validationError < validationErrorBest:
        validationErrorBest = validationError
        '''
        with open('netBestSAnoSAattempt1netA.pickle' , 'wb') as handle:
            print('saving the modelA....')
            pickle.dump(networkAOut , handle)
        '''
        with open('netBestSAnoSAattempt1netB.pickle' , 'wb') as handle:
            print('saving the modelB....')
            pickle.dump(networkBOut , handle)

    '''
    TestX , TestY = load_test_data()

    TestY = np.squeeze(TestY)
    TestErr , Testaccuracy = valid_func(TestX, TestY)
    '''
    print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))
    #print("  test accuracy:\t\t{:.2f} %".format(
     #   Testaccuracy * 100))

    TRAINLOSS = np.append(TRAINLOSS , train_err / train_batches)
    VALIDATIONLOSS = np.append(VALIDATIONLOSS , val_err / val_batches)
    VALIDATIONACC = np.append(VALIDATIONACC , val_acc / val_batches)

    #TESTACC = np.append(TESTACC , Testaccuracy)

    scipy.io.savemat('LossesSAnoSAattempt1.mat',mdict={'TrainLoss':TRAINLOSS , 'ValidationLoss':VALIDATIONLOSS,'ValidatioAcc':VALIDATIONACC})
print 'yey'