import numpy as np
import lmdb
import caffe as c
import sys

featDim = 26752 
labDim = 26752 
mbSize = 64
if len(sys.argv) > 1:
	mbSize = int(sys.argv[1])
totalCount = mbSize * 40

features = np.random.randn(totalCount, 1, 1, featDim)
labels = np.random.randint(0, labDim, size=(totalCount,))

db = lmdb.open('./fake_data26752.lmdb', map_size=features.nbytes * 40)

with db.begin(write = True) as txn:
  for i in range(totalCount):
    d = c.proto.caffe_pb2.Datum()
    d.channels = features.shape[1]
    d.height = features.shape[2]
    d.width = features.shape[3]
    d.data = features[i].tostring()
    d.label = labels[i]
    txn.put('{:08}'.format(i), d.SerializeToString())

