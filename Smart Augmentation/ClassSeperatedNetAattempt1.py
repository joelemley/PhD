import numpy as np
import scipy.io
import time
import pickle
import lasagne
import theano.tensor as T
import theano
import sys
import random

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
validate1=np.reshape(validate1,(100,256,256,3))
validate1=np.transpose((validate1),(0,3,1,2))



validate2 = np.fromfile('valclass2',dtype='uint8')
validate2 = np.reshape(validate2,(100,3,256,256))
validate2=np.reshape(validate2,(100,256,256,3))
validate2=np.transpose((validate2),(0,3,1,2))


imageW=256
imageH=256
TrainData = {'imClass1':train1,'imClass2':train2}
ValidationData = {'imClass1':validate1,'imClass2':validate2}

#Training Parameters

num_epochs = 1000
validationErrorBest = 100000;
validationAccuracyBest = 0;
alpha = .7
beta = .3
netAnumInputs = 2
batchSize = 5


numSamplesTrain=len(TrainData["imClass1"])+len(TrainData["imClass2"])
numSamplesVal=len(ValidationData["imClass1"])+len(ValidationData["imClass2"])

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
def Give_Right_Data_DoubleNet(TrainValData, batchSize, NetAinputNums):
    DataX1 = np.empty(shape=[batchSize, NetAinputNums*3, imageW, imageH])
    DataX2 = np.empty(shape=[batchSize, NetAinputNums*3, imageW, imageH])
    DataY1 = np.empty(shape=[batchSize, 3, imageW, imageH])
    DataY2 = np.empty(shape=[batchSize, 3, imageW, imageH])

    # Class1
    for ii in range(batchSize):
        samplePerm1 = np.random.permutation(TrainValData['imClass1'].shape[0])
        tempBatchX1 = TrainValData['imClass1'][samplePerm1[0:NetAinputNums], :, :, :]
        tempBatchY1 = TrainValData['imClass1'][samplePerm1[NetAinputNums:NetAinputNums + 1], :, :, :]
        for jj in range(NetAinputNums):
            DataX1[ii : (ii+1) , jj*3 : (jj+1)*3,:,:]=tempBatchX1[jj,:,:,:]
        DataY1[ii:ii+1,:,:,:] = tempBatchY1
    # Class2
    for ii in range(batchSize):
        samplePerm2 = np.random.permutation(TrainValData['imClass2'].shape[0])
        tempBatchX2 = TrainValData['imClass2'][samplePerm2[0:NetAinputNums], :, :, :]
        tempBatchY2 = TrainValData['imClass2'][samplePerm2[NetAinputNums:NetAinputNums + 1], :, :, :]
        for jj in range(NetAinputNums):
            DataX2[ii : (ii+1) , jj*3 : (jj+1)*3,:,:]=tempBatchX2[jj,:,:,:]
        DataY2[ii:ii+1,:,:,:] = tempBatchY2

    target1 = np.zeros(shape=(batchSize*2 , 1),dtype=np.uint8)
    target2 = np.ones(shape=(batchSize*2 , 1),dtype=np.uint8)
    target3 = np.concatenate((target1 , target2))

    return np.float32(DataX1)/np.float32(255) , np.float32(DataY1)/np.float32(255) , np.float32(DataX2)/np.float32(255) , np.float32(DataY2)/np.float32(255) ,np.uint8(target1),np.uint8(target2), np.uint8(target3)



def Give_Right_Data_DoubleNet_validation_test(TrainValData, batchSize, NetAinputNums):
    DataX1 = np.empty(shape=[batchSize, NetAinputNums*3, imageW, imageH])
    DataX2 = np.empty(shape=[batchSize, NetAinputNums*3, imageW, imageH])
    DataY1 = np.empty(shape=[batchSize, 3, imageW, imageH])
    DataY2 = np.empty(shape=[batchSize, 3, imageW, imageH])

    # Class1
    for ii in range(batchSize):
        samplePerm1 = np.random.permutation(TrainValData['imClass1'].shape[0])
        tempBatchX1 = TrainValData['imClass1'][samplePerm1[0:NetAinputNums], :, :, :]
        tempBatchY1 = TrainValData['imClass1'][samplePerm1[NetAinputNums:NetAinputNums + 1], :, :, :]
        for jj in range(NetAinputNums):
            DataX1[ii: (ii + 1), jj*3: (jj + 1)*3, :, :] = tempBatchX1[jj, :, :, :]
        DataY1[ii:ii + 1, :, :, :] = tempBatchY1
    # Class2
    for ii in range(batchSize):
        samplePerm2 = np.random.permutation(TrainValData['imClass2'].shape[0])
        tempBatchX2 = TrainValData['imClass2'][samplePerm2[0:NetAinputNums], :, :, :]
        tempBatchY2 = TrainValData['imClass2'][samplePerm2[NetAinputNums:NetAinputNums + 1], :, :, :]
        for jj in range(NetAinputNums):
            DataX2[ii: (ii + 1), jj*3: (jj + 1)*3, :, :] = tempBatchX2[jj, :, :, :]
        DataY2[ii:ii + 1, :, :, :] = tempBatchY2

    target1 = np.zeros(shape=(batchSize, 1), dtype=np.uint8)
    target2 = np.ones(shape=(batchSize, 1), dtype=np.uint8)
    target3 = np.concatenate((target1, target2))

    return np.float32(DataX1)/np.float32(255), np.float32(DataY1)/np.float32(255), np.float32(DataX2)/np.float32(255), np.float32(DataY2)/np.float32(255), np.uint8(target3)


print('Building the model...')
input_varA1 = T.tensor4('inputsA1')
input_varA2 = T.tensor4('inputsA2')
input_varB = T.tensor4('inputsB')
target_varA1 = T.tensor4('targetsA1')
target_varA2 = T.tensor4('targetsA2')
target_varB1 = T.ivector('targetsB1')
target_varB2 = T.ivector('targetsB2')
target_varB3 = T.ivector('targetsB3')
target_varB = T.ivector('targetsB')
flags1tensor = T.tensor4('flags1')
flags2tensor = T.tensor4('flags2')

#Network A1

networkA1i = lasagne.layers.InputLayer(shape=(None , netAnumInputs*3 , None , None) , name='networkA1i')
netA1L1 = lasagne.layers.Conv2DLayer(networkA1i,num_filters=16,filter_size=(3,3),pad='same',name='netA1L1')

netA1L2 = lasagne.layers.Conv2DLayer(netA1L1 , num_filters=16,filter_size=(5,5),pad='same',name='netA1L2')

netA1L3 = lasagne.layers.Conv2DLayer(netA1L2,num_filters=32,filter_size=(7,7),pad='same',name='netA1L3')

netA1L4 = lasagne.layers.Conv2DLayer(netA1L3 , num_filters=32,filter_size=(5,5),pad='same',name='netA1L4')

networkA1Out = lasagne.layers.Conv2DLayer(netA1L4,num_filters=3,filter_size=(3,3),pad='same',name='networkA1Out')

#Network A2
networkA2i = lasagne.layers.InputLayer(shape=(None , netAnumInputs*3 , None , None) , name='networkA2i')
netA2L1 = lasagne.layers.Conv2DLayer(networkA2i,num_filters=16,filter_size=(3,3),pad='same',name='netA2L1')

netA2L2 = lasagne.layers.Conv2DLayer(netA2L1 , num_filters=16,filter_size=(5,5),pad='same',name='netA2L2')

netA2L3 = lasagne.layers.Conv2DLayer(netA2L2,num_filters=32,filter_size=(7,7),pad='same',name='netA2L3')

netA2L4 = lasagne.layers.Conv2DLayer(netA2L3 , num_filters=32,filter_size=(5,5),pad='same',name='netA2L4')

networkA2Out = lasagne.layers.Conv2DLayer(netA2L4,num_filters=3,filter_size=(3,3),pad='same',name='networkA2Out')


#Network B

networkBi = lasagne.layers.InputLayer(shape=(None,3,imageW, imageH),name='networkBi')

netBL1 = lasagne.layers.Conv2DLayer(networkBi,num_filters=16,filter_size=(3,3),name='netBL1')
netBIBN = batch_norm(netBL1,name='netBIBN')
netBL1MP = lasagne.layers.MaxPool2DLayer(netBIBN , pool_size=(3,3),name='netBL1MP')

netBL2 = lasagne.layers.Conv2DLayer(netBL1MP , num_filters=8,filter_size=(3,3),name='netBL2')
netBIBN2 = batch_norm(netBL2,name='netBIBN2')
netBL2MP = lasagne.layers.MaxPool2DLayer(netBIBN2 , pool_size=(3,3),name='netBL2MP')

netBL3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL2MP),num_units=1024,name='netBL3')
networkBOut = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL3),num_units=2,nonlinearity=lasagne.nonlinearities.softmax,name='networkBOut')

# Train path
outputA1 = lasagne.layers.get_output(networkA1Out, inputs=input_varA1)
lossA1 = lasagne.objectives.squared_error(outputA1, target_varA1)
lossA1 = lossA1.mean()

outputA2 = lasagne.layers.get_output(networkA2Out , inputs=input_varA2)
lossA2 = lasagne.objectives.squared_error(outputA2 , target_varA2)
lossA2 = lossA2.mean()

# path 1 A1|a2->B
netBin1 = T.concatenate((outputA1, target_varA1 , outputA2 , target_varA2))
outputB1 = lasagne.layers.get_output(networkBOut , inputs=netBin1)
lossB1 = lasagne.objectives.categorical_crossentropy(outputB1, target_varB3)
lossB1 = lossB1.mean()
loss_total1 = alpha*lossA1 + beta*lossB1

paramsA1 = lasagne.layers.get_all_params(networkA1Out , trainable = True)
paramsB = lasagne.layers.get_all_params(networkBOut , trainable = True)
params_total1  = list(np.concatenate((paramsA1 ,  paramsB)))
updates1 = lasagne.updates.nesterov_momentum(loss_total1 , params_total1 , learning_rate=.0005, momentum=.5)

# path 2 a1||A2->B
loss_total2 = alpha*lossA2 + beta*lossB1
paramsA2 = lasagne.layers.get_all_params(networkA2Out , trainable = True)
#paramsB2 = lasagne.layers.get_all_params(networkBOut , trainable = True)
params_total2  = list(np.concatenate((paramsA2 ,  paramsB)))
updates2 = lasagne.updates.nesterov_momentum(loss_total2 , params_total2 , learning_rate=.0005, momentum=.5)


# path3 A1||A2->B
loss_total3 = alpha*(lossA1 + lossA2) + beta*lossB1
params_total3 = list(np.concatenate((paramsA1,paramsA2,paramsB)))
updates3 = lasagne.updates.nesterov_momentum(loss_total3, params_total3 , learning_rate=.0005, momentum=.5)

#Merging pathes
updates = updates1.copy()
updates.update(updates2)

updates[netBIBN.beta] = updates3[netBIBN.beta]
updates[netBIBN.gamma] = updates3[netBIBN.gamma]
updates[netBIBN2.beta] = updates3[netBIBN2.beta]
updates[netBIBN2.gamma] = updates3[netBIBN2.gamma]
updates[netBL1.W] = updates3[netBL1.W]
updates[netBL2.W] = updates3[netBL2.W]
updates[netBL3.W] = updates3[netBL3.W]
updates[netBL3.b] = updates3[netBL3.b]
updates[networkBOut.W] = updates3[networkBOut.W]
updates[networkBOut.b] = updates3[networkBOut.b]




# Test Path B
test_outputB = lasagne.layers.get_output(networkBOut,inputs=input_varB,deterministic=True)
test_lossB = lasagne.objectives.categorical_crossentropy(test_outputB,target_varB)
test_lossB = test_lossB.mean()
test_acc = T.mean(T.eq(T.argmax(test_outputB, axis=1), target_varB),
                  dtype=theano.config.floatX)

train_func = theano.function([input_varA1 , target_varA1 , input_varA2 , target_varA2 ,target_varB3] , [loss_total3,lossB1] , updates=updates)
valid_func = theano.function([input_varB , target_varB] , [test_lossB , test_acc])

print('Training...')
TRAINLOSS = np.array([])
VALIDATIONLOSS = np.array([])
VALODATIONACC = np.array([])
TestAccuracy=np.array([])
TestError=np.array([])
TRAINERRORB = np.array([])

e=0

for epoch in range(num_epochs):
    e=e+1
    train_err = 0
    train_batches = 0
    train_errB =0
    start_time = time.time()
    for kk in range(int(numSamplesTrain/batchSize)):
        inputsA1 , targetsA1 , inputsA2 , targetsA2,targetsB1,targetsB2 , targetsB3 = Give_Right_Data_DoubleNet(TrainData,batchSize=batchSize,NetAinputNums=netAnumInputs)
        targetsB1 = np.squeeze(targetsB1)
        targetsB2 = np.squeeze(targetsB2)
        targetsB3 = np.squeeze(targetsB3)
        train_errtemp , trainErrorBtemp = train_func(inputsA1 , targetsA1 , inputsA2 , targetsA2, targetsB3)
        train_err += train_errtemp
        train_errB += trainErrorBtemp
        train_batches += 1
        #err = valid_func(inputs , targets)
    val_acc = 0
    val_err = 0
    val_batches = 0

    for pp in range(int(numSamplesVal/batchSize)):
        inputsA1 , targetsA1 , inputsA2 , targetsA2 , targetsB3 = Give_Right_Data_DoubleNet_validation_test(ValidationData,batchSize=batchSize*2,NetAinputNums=netAnumInputs)
        targetsB3 = np.squeeze(targetsB3)
        err , acc  = valid_func(np.concatenate((targetsA1 , targetsA2)) , targetsB3)
        val_err += err
        val_acc += acc
        val_batches +=1

    validationError = val_err / val_batches
    validationAccuracy=val_acc/val_batches

    if validationError < validationErrorBest:
        validationErrorBest = validationError
        with open('netBestMNAttempt1SAnetA1err.pickle' , 'wb') as handle:
            print('saving the modelA1....')
            pickle.dump(networkA1Out , handle)
        with open('netBestMNAttempt1SAnetA2err.pickle' , 'wb') as handle:
            print('saving the modelA2....')
            pickle.dump(networkA2Out , handle)
        with open('netBestMNAttempt1SAnetBerr.pickle' , 'wb') as handle:
            print('saving the modelB....')
            pickle.dump(networkBOut , handle)

    # if validationAccuracy > validationAccuracyBest:
    #     validationAccuracyBest=validationAccuracy
    #     with open('netBestAttempt3SAnetAacc'+'.pickle', 'wb') as handle:
    #         print('saving the modelA....')
    #         pickle.dump(networkAOut, handle)
    #     with open('netBestAttempt3SAnetBacc'+'.pickle', 'wb') as handle:
    #         print('saving the modelB....')
    #         pickle.dump(networkBOut, handle)
    '''
    TestData = scipy.io.loadmat('TestingData.mat')

    TestX = np.concatenate((TestData['imClass1'],TestData['imClass2']),axis=0)
    TestY = np.concatenate((np.zeros(shape=(len(TestData['imClass1']) , 1)) , np.ones(shape=(len(TestData['imClass2']) , 1))))

    TestX = TestX.reshape(-1,1,96,96)

    TestX=np.float32(TestX)
    TestY= np.uint8(TestY)
    TestY = np.squeeze(TestY)
    Testerr, Testacc = valid_func(TestX,TestY)
    '''
    print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  training loss netB:\t\t{:.6f}".format(train_errB / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))
    #print "Test Accuracy: "+str(Testacc)
    TRAINLOSS = np.append(TRAINLOSS , train_err / train_batches)
    VALIDATIONLOSS = np.append(VALIDATIONLOSS , val_err / val_batches)
    #TestError=np.append(TestError,Testerr)
    #TestAccuracy = np.append(TestAccuracy, Testacc)
    TRAINERRORB = np.append(TRAINERRORB,train_errB / train_batches)


    plt.plot(range(len(TRAINLOSS)),TRAINLOSS, color='r', label="Train Loss")
    plt.plot(range(len(VALIDATIONLOSS)),VALIDATIONLOSS, color='g',label="Validation Loss")
    if(e==1):
        plt.legend()
    plt.show()
    plt.pause(0.0001)
    scipy.io.savemat('LossesMNAttempt1SA.mat',mdict={'TrainLoss':TRAINLOSS , 'ValidationLoss':VALIDATIONLOSS, 'TrainLossB':TRAINERRORB})

























print 'yuuo'