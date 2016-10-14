# deep-learning-benchmarks

### Hardware Config

| Name | Value |
|------|-------|
| CPU	| Intel i7 5960X, 8 Cores @ 3.00GHZ | 
| GPU	| GTX Titan X(Maxwell) | 
| Memory |	DDR4, 8 * 16GB @ 2133MHZ |
| Hard Drive |	256GB SSD | 

### Software Config

| Name | Value |
|------|-------|
|OS	| Ubuntu 14.04 LTS |
|GPU | driver	367.44 |
|CUDA	| 7.5 |
|cuDNN |	V5.1 |


### Benchmark Network Config

|Name | Batch Size | Input | Output | Layers | #Parameters |
|-----|------------|------|------|------|-------|
| FCN-5 | 8192 | 512 | 1000 | 5 | ~11 millions |
| FCN-8 | 8192 | 512 | 1000 | 8 | ~24millions |
| AlexNet | 16 | 150528 | 1000 | 4 | ~62 millions |
| ResNet-50 | 16 | 150528 | 1000 | 50 | ~25 millions |
| LSTM-32 | 128 | 10000 | 10000 | 2 | ~6 millions |
| LSTM-64 | 128 | 10000 | 10000 | 2 | ~6 millions |

### Tool version

| Tool     | Version |
|----------|---------|
| Caffe | [d208b71](https://github.com/BVLC/caffe/commit/d208b714abb8425f1b96793e04508ad21724ae3f) |
| CNTK |[v1.7.1](https://cntk.ai/dll1-1.7.1.html)|
| MXNet | [2d0e3ac](https://github.com/dmlc/mxnet/tree/2d0e3ac8f017b15abf171f7acf0a3631fc4e2970)
| TensorFlow |  [r0.10](https://github.com/tensorflow/tensorflow/tree/r0.10) |
| Theano | [0.8.X](https://github.com/Theano/Theano/tree/0.8.X) |
| Torch | [95f137f](https://github.com/torch/torch7/tree/95f137f635c3b01d89b9c008b68a9321ca28e59b) |


### References

https://github.com/n17s/hkbu-benchmark

https://github.com/FreemanX/hkbu-benchmark



Disclaimer: I'm a Microsoft employee, however, this is my personal github account and information/code shared here does not represent opinions or views of Microsoft in any way.
