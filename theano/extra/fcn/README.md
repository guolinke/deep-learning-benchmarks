
# About the theano fcn
Two other version of fcn5 with keras and raw theano have been implemented to verify that the lasagne version fcn5 is doing the right thing.

The lasagne and raw theano version are much faster than the keras version.
The raw theano versio is a little slower than than lasagne version because  lasagne version uses an int vector as the label instead of a one-hot matrix, otherwise both should perform same on speed.


Two profiling tools are used to check what the GPU has done.
* Theano profiling tools : We find lasagne and the raw theano version almost called the same GPU function.  But the keras version does some exstra  calls.   An operation called ` GpuElemwise` brings too much extra overhead. We are still investigating the reason.
* Nvprof :  The result shows similar information like theano profiling tools.
