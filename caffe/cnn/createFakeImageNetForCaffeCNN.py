import numpy as np
import lmdb
import caffe as c
import sys

mbSize = 16
if len(sys.argv) > 1:
	mbSize = int(sys.argv[1])
totalCount = mbSize * 200

features = np.random.randn(totalCount, 3, 224, 224)
labels = np.random.randint(0, 1000, size=(totalCount,))

db = lmdb.open('./fake_image_net.lmdb', map_size=features.nbytes * 10)

with db.begin(write = True) as txn:
  for i in range(totalCount):
    d = c.proto.caffe_pb2.Datum()
    d.channels = features.shape[1]
    d.height = features.shape[2]
    d.width = features.shape[3]
    d.data = features[i].tostring()
    d.label = labels[i]
    txn.put('{:08}'.format(i), d.SerializeToString())




