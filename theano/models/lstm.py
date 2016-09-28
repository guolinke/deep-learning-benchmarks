import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, EmbeddingLayer, ReshapeLayer
import numpy as np




def build_function(vocab_size=10000, seq=32, hiddenLayDim=256):
    embedding_size = 256
    featureDim = (seq, )
    labelDim = vocab_size

    def build_model(batch_size=128):
        x = T.matrix('input', dtype='int32')

        l_in = lasagne.layers.InputLayer(shape=(None, None,), input_var=x)
        # We can retrieve symbolic references to the input variable's shape, which
        # we will later use in reshape layers.
        batchsize, seqlen = l_in.input_var.shape

        W = np.random.rand(vocab_size, embedding_size).astype(np.float32)
        ebd = EmbeddingLayer(l_in, input_size=vocab_size, output_size=embedding_size, W=W)

        # All gradients above this will be clipped
        GRAD_CLIP = 100
        # We now build the LSTM layer which takes l_in as the input layer
        # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

        l_forward_1 = lasagne.layers.LSTMLayer(
            ebd, hiddenLayDim, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)

        l_forward_2 = lasagne.layers.LSTMLayer(
            l_forward_1, hiddenLayDim, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)

        # the output size of l_forward_2 will be (batch_size, seqlen, hiddenLayDim)

        # In order to connect a recurrent layer to a dense layer, we need to
        # flatten the first two dimensions (batch_size, seqlen); this will
        # cause each time step of each sequence to be processed independently
        l_shp = ReshapeLayer(l_forward_2, (-1, hiddenLayDim))
        l_out = lasagne.layers.DenseLayer(l_shp, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=None)
        # Don't reshape back. Because keep the current shape will make it
        # easier to calc the categorical_crossentropy
        # l_out = ReshapeLayer(l_dense, (batchsize, seqlen, vocab_size))
        return l_out, x

    def input_generator(batch_size):
        return np.random.randint(vocab_size,
            size=(batch_size,) + featureDim).astype(np.int32)

    def get_output_size(batch_size):
        return (batch_size * seq, )

    return build_model, featureDim, labelDim, input_generator, get_output_size


build_model32, featureDim32, labelDim32, input_generator32, get_output_size32 = build_function(seq=32)
build_model64, featureDim64, labelDim64, input_generator64, get_output_size64 = build_function(seq=64)
