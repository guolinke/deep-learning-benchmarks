# deep-learning-benchmarks

### Benchmark Environment:

OS: Ubuntu 14.04 LTS

CPU: Intel i7 5960X, 8 Cores @ 3.00GHZ

Memory: DDR4, 8 * 16GB @ 2133MHZ

GPU: GTX Titan X(Maxwell), Driver version: 367.27, CUDA 7.5, cuDNN v5.1

Hard Drive: 256GB SSD


### Benchmark Results

| Tool     | FCN-5 | FCN-8 | AlexNet | ResNet | LSTM-32 | LSTM-64 |
|-----------|-------|-------|---------|--------|---------|---------|
| Caffe | 0.023 | 0.025 | 0.040   | 0.265  |  n/a    | n/a     |
| CNTK  | 0.030 | 0.035 |  0.036  | 0.186  | 0.033   | 0.061   |
|TensorFlow| 0.054 | 0.057 | 0.031   | 0.250  |  0.078  | 0.137   |
| Theano | 0.071 | 0.076 | 0.032   | 0.597  |  0.103  |  0.194  |
| Torch  | 0.026 | 0.028 | 0.024   | 0.159  |  0.100  | 0.200   |

### References

https://github.com/n17s/hkbu-benchmark

https://github.com/FreemanX/hkbu-benchmark
