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
|GPU | driver	367.27 |
|CUDA	| 7.5 |
|cuDNN |	V5.1 |


### Benchmark Network Config

|Name | Input | Output | Layers | #Parameters |
|---|------|------|------|------|
| FCN-5 | 26752 | 26752 | 5 | ~118 millions |
| FCN-8 | 26752 | 26752 | 8 | ~131 millions |
| AlexNet | 150528 | 1000 | 4 | ~62 millions |
| ResNet-50 | 150528 | 1000 | 50 | ~25 millions |
| LSTM-32 | 10000 | 10000 | 2 | ~6 millions |
| LSTM-64 | 10000 | 10000 | 2 | ~6 millions |

### Tool version

| Tool     | Version |
|----------|---------|
| Caffe | [Link](https://github.com/BVLC/caffe/tree/7f8f9e146d90172e457678866961b86ae4218824) |
| CNTK |[Link](https://github.com/Microsoft/CNTK/tree/cac191c8c3c08e546c9af25236d368c0ed2812c2)|
| TensorFlow |  [link](https://github.com/tensorflow/tensorflow/tree/bc64f05d4090262025a95438b42a54bfdc5bcc80) |
| Theano | [Link](https://github.com/Theano/Theano/tree/140d0a064523349b630a284247c7cddd767fc46e) |
| Torch | [Link](https://github.com/torch/torch7/tree/95f137f635c3b01d89b9c008b68a9321ca28e59b) |

### Benchmark Results


seconds/num_batches:

| Tool | FCN-5 | FCN-8 | AlexNet | ResNet | LSTM-32 | LSTM-64 |
|------|-------|-------|---------|--------|---------|---------|
|Caffe| 0.023 | 0.025 | 0.040 | 0.266 | n/a | n/a |
|CNTK| 0.031 | 0.035 | 0.037 | 0.187 | 0.033 | 0.061 |
|TensorFlow| 0.055 | 0.057 | 0.031 | 0.251 | 0.077 | 0.138 |
|Theano| 0.038 | 0.043 | 0.037 | 0.598 | 0.042 | 0.079 |
|Torch| 0.026 | 0.028 | 0.024 | 0.159 | 0.104 | 0.206 |
samples/second:

| Tool | FCN-5 | FCN-8 | AlexNet | ResNet | LSTM-32 | LSTM-64 |
|------|-------|-------|---------|--------|---------|---------|
|Caffe| 2780 | 2529 | 395 | 60 | n/a | n/a |
|CNTK| 2086 | 1853 | 438 | 85 | 3863 | 2088 |
|TensorFlow| 1172 | 1120 | 516 | 63 | 1659 | 929 |
|Theano| 1684 | 1488 | 432 | 26 | 3047 | 1620 |
|Torch| 2424 | 2248 | 674 | 100 | 1233 | 621 |
### References

https://github.com/n17s/hkbu-benchmark

https://github.com/FreemanX/hkbu-benchmark
