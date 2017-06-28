# deep-learning-benchmarks

***Note: this benchmark is out of date. ***

### Hardware Config

| Name | Value |
|------|-------|
| CPU	| E5-2680 v2 | 
| GPU	| K40m | 
| Memory |	DDR3, 256GB @ 1600Mhz |

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
| Caffe | [4ba654f](https://github.com/BVLC/caffe/tree/4ba654f5c88c36ee8ba53964b7faf25c6d7010b4) |
| CNTK |[v1.7.1](https://cntk.ai/dll1-1.7.1.html)|
| MXNet | [32cb6bc](https://github.com/dmlc/mxnet/tree/32cb6bc0a95fb351763ad82f1deca8a9024d5027)
| TensorFlow |  [r0.10](https://github.com/tensorflow/tensorflow/tree/r0.10) |
| Theano | [0.8.X](https://github.com/Theano/Theano/tree/0.8.X) |
| Torch | [95f137f](https://github.com/torch/torch7/tree/95f137f635c3b01d89b9c008b68a9321ca28e59b) |


### References

https://github.com/n17s/hkbu-benchmark

https://github.com/FreemanX/hkbu-benchmark



Disclaimer: I'm a Microsoft employee, however, this is my personal github account and information/code shared here does not represent opinions or views of Microsoft in any way.
