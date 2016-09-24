# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import numpy as np
import mxnet as mx
import argparse
from lstm import lstm_unroll
from bucket_io import BucketSentenceIter, default_build_vocab

def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

# Argument
parser=argparse.ArgumentParser(description='choose model to test benchmark')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-batch', type=int, default=100)
parser.add_argument('--printmsg', type=bool, default=True)
parser.add_argument('--num-embed', type=int, default=256)
parser.add_argument('--num-hidden', type=int, default=256)
parser.add_argument('--lstm-layer', type=int, default=2)
parser.add_argument('--seq-len', type=int, default=32)
parser.add_argument("--data-path", default="")
args=parser.parse_args()

# Dump out msg
if args.printmsg == True:
    print '************** Config Message **************'
    print 'model: LSTM-'+str(args.seq_len)
    print 'gpu: '+str(args.gpu)
    print 'Mini_batch_size: '+str(args.batch_size)
    #print 'Num_batch: '+str(args.num_batch)
    print 'Num_hidden: '+str(args.num_hidden)
    print 'Num_embed: '+str(args.num_embed)
    print 'Num_lstm_layer: '+str(args.lstm_layer)
    print '********************************************'

if __name__ == '__main__':
    batch_size = args.batch_size
    buckets = [args.seq_len]
    num_hidden = args.num_hidden
    num_embed = args.num_embed
    num_lstm_layer = args.lstm_layer

    num_epoch = 1
    learning_rate = 0.01
    momentum = 0.0

    # dummy data is used to test speed without IO
    dummy_data = False

    contexts = [mx.context.gpu(i) for i in range(1)]

    vocab = default_build_vocab(args.data_path+"/ptb.train.txt")

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, 10000,
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label=10000)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = BucketSentenceIter(args.data_path+"/ptb.train.txt", vocab,
                                    buckets, batch_size, init_states)
    data_val = BucketSentenceIter(args.data_path+"/ptb.valid.txt", vocab,
                                  buckets, batch_size, init_states)

    if dummy_data:
        data_train = DummyIter(data_train)
        data_val = DummyIter(data_val)

    if len(buckets) == 1:
        # only 1 bucket, disable bucketing
        symbol = sym_gen(buckets[0])
    else:
        symbol = sym_gen
        
    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    import time
    tic = time.time()
    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Perplexity),
              batch_end_callback=None)
    toc = time.time()
    elasped_time = float(toc - tic)
    print '********************** Training on GPU() **********************'
    print 'Avg elasped time per mini-batch (sec/mini-batch): '+str(round(elasped_time/(float(data_train.sample[args.seq_len]) / (batch_size * 32)), 6))
    print 'Avg samples per second (samples/sec): '+str(int(round((data_train.sample[args.seq_len])/elasped_time)))
    print '****************************************************************'
    # Parameter counting
    print 'Parameter Number: '+str(param_count(arg_names, arg_map))
