import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, EmbeddingLayer
import numpy as np




def build_function(vocab_size=10000, seq=32, hiddenLayDim=256):
    embedding_size = 256
    featureDim = (seq, )
    labelDim = vocab_size

    def build_model(batch_size=128):
        x = T.matrix('input', dtype='int32')

        l_in = lasagne.layers.InputLayer(shape=(None, None,), input_var=x)

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
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True)

        # The output of l_forward_2 of shape (batch_size, hiddenLayDim) is then passed through the softmax nonlinearity to
        # create probability distribution of the prediction
        # The output of this stage is (batch_size, vocab_size)
        l_out = lasagne.layers.DenseLayer(l_forward_2, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=None)
        return l_out, x

    def input_generator(*dims):
        return np.random.randint(vocab_size, size=dims).astype(np.int32)

    return build_model, featureDim, labelDim, input_generator


build_model32, featureDim32, labelDim32, input_generator32 = build_function(seq=32)
build_model64, featureDim64, labelDim64, input_generator64 = build_function(seq=64)
