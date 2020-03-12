import numpy as np
import scipy.io

import time
import pickle

import matplotlib.pyplot as plt


TrainData=scipy.io.loadmat("TrainingData")
ValidationData=scipy.io.loadmat("ValidationData")
#TestData=scipy.io.loadmat("TestingData")

numSamplesTrain = len(TrainData['imClass1'])+len(TrainData['imClass2'])
numSamplesValidation = len(ValidationData['imClass1'])+len(ValidationData['imClass2'])

import lasagne
import theano.tensor as T
import theano


class Unpool2DLayer(lasagne.layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(ds[0], axis=2).repeat(ds[1], axis=3)


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

def load_test_data():

    TestData = scipy.io.loadmat('TestingData.mat')

    TestX = np.concatenate((TestData['imClass1'],TestData['imClass2']),axis=0)
    TestY = np.concatenate((np.zeros(shape=(len(TestData['imClass1']) , 1)) , np.ones(shape=(len(TestData['imClass2']) , 1))))

    TestX = TestX.reshape(-1,1,96,96)

    return np.float32(TestX) , np.uint8(TestY)


def Give_Right_Data_SA_2(TrainValData , batchSize , NetAinputNums):#classPerm = np.random.permutation(10)
#batchSize = 100
#NetAinputNums = 5
    DataX = np.empty(shape=[batchSize,  NetAinputNums, 96, 96])
    DataY = np.empty(shape=[batchSize, 1, 96, 96])
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
            DataX[ii: (ii + 1), jj : (jj + 1) , :, :] = tempBatchX[jj, :, :, :]
        DataY[ii: ii + 1, :, :, :] = tempBatchY
    return np.float32(DataX) , np.float32(DataY) , np.uint8(np.concatenate((classPerm,classPerm)))

def Give_Right_Data_SA(TrainValData , batchSize , NetAinputNums):#classPerm = np.random.permutation(10)
#batchSize = 100
#NetAinputNums = 5
    DataX = np.empty(shape=[batchSize,  NetAinputNums, 96, 96])
    DataY = np.empty(shape=[batchSize, 1, 96, 96])
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
            DataX[ii: (ii + 1), jj : (jj + 1) , :, :] = tempBatchX[jj, :, :, :]
        DataY[ii: ii + 1, :, :, :] = tempBatchY
    return np.float32(DataX) , np.float32(DataY) , np.uint8(classPerm)
alpha = .5
beta = .5
netAnumInputs = 5
batchSize = 50
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
networkBi = lasagne.layers.InputLayer(shape=(None,1,96,96),name='networkBi')
netBL1 = lasagne.layers.Conv2DLayer(networkBi,num_filters=16,filter_size=(3,3),name='netBL1')
netBL1BN = batch_norm(netBL1 , name='netBL1BN')
netBL1MP = lasagne.layers.MaxPool2DLayer(netBL1BN , pool_size=(3,3),name='netBL1MP')
netBL2 = lasagne.layers.Conv2DLayer(netBL1 , num_filters=8,filter_size=(3,3),name='netBL2')
netBL2BN = batch_norm(netBL2 , name='netBL2BN')
netBL2MP = lasagne.layers.MaxPool2DLayer(netBL2BN , pool_size=(3,3),name='netBL2MP')
netBL3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL2MP),num_units=1024,name='netBL3')
networkBOut = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL3),num_units=2,nonlinearity=lasagne.nonlinearities.softmax,name='networkBOut')

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

updates = lasagne.updates.nesterov_momentum(loss_total , params_total , learning_rate=.01,momentum=0.9)
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
        with open('netBestAttempt3SAnetA.pickle' , 'wb') as handle:
            print('saving the modelA....')
            pickle.dump(networkAOut , handle)
        '''
        with open('netBestAttempt3SAnoSA.pickle' , 'wb') as handle:
            print('saving the modelB....')
            pickle.dump(networkBOut , handle)
    TestX , TestY = load_test_data()

    TestY = np.squeeze(TestY)
    TestErr , Testaccuracy = valid_func(TestX, TestY)
    print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))
    print("  test accuracy:\t\t{:.2f} %".format(
        Testaccuracy * 100))

    TRAINLOSS = np.append(TRAINLOSS , train_err / train_batches)
    VALIDATIONLOSS = np.append(VALIDATIONLOSS , val_err / val_batches)
    VALIDATIONACC = np.append(VALIDATIONACC , val_acc / val_batches)
    TESTACC = np.append(TESTACC , Testaccuracy)
    scipy.io.savemat('LossesAttempt3SAnoSA.mat',mdict={'TrainLoss':TRAINLOSS , 'ValidationLoss':VALIDATIONLOSS,'TestAcc':TESTACC,'ValidatioAcc':VALIDATIONACC})
print 'yey'
print 'yey'
