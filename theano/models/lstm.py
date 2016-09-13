import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer




def build_function(vocab_size=10010, seq=32, hiddenLayDim=256):
    featureDim = (seq, vocab_size)
    labelDim = vocab_size

    def build_model(batch_size=128):
        x = T.tensor3('input')

        l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size), input_var=x)

        # All gradients above this will be clipped
        GRAD_CLIP = 100
        # We now build the LSTM layer which takes l_in as the input layer
        # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

        l_forward_1 = lasagne.layers.LSTMLayer(
            l_in, hiddenLayDim, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)

        l_forward_2 = lasagne.layers.LSTMLayer(
            l_forward_1, hiddenLayDim, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True)

        # The output of l_forward_2 of shape (batch_size, hiddenLayDim) is then passed through the softmax nonlinearity to
        # create probability distribution of the prediction
        # The output of this stage is (batch_size, vocab_size)
        l_out = lasagne.layers.DenseLayer(l_forward_2, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)
        return l_out, x
    return build_model, featureDim, labelDim


build_model32, featureDim32, labelDim32 = build_function(seq=32)
build_model64, featureDim64, labelDim64 = build_function(seq=64)
