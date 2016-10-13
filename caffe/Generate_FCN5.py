#!/usr/bin/python

import sys

batch_size = int(sys.argv[1])
if len(sys.argv) > 2:
	# for multi gpu
	batch_size /= int(sys.argv[2])

output = open('fcn/fcn5.prototxt','w')
output_solver = open('fcn/fcn5-solver.prototxt','w')

out_solver_str = '''
base_lr: 0.01
lr_policy: "fixed"
max_iter: 40 
display: 1
#solver_mode: GPU
net: "fcn5.prototxt"
solver_mode: GPU

'''

out_str = '''name: "FCN5"  
layer {  
  name: "data"  
  type: "Data"  
  top: "data"  
  top: "label"  
  include {  
    phase: TRAIN  
  }  
  data_param: {  
    batch_size: %d   
    source: "fake_data26752.lmdb"  
    backend: LMDB  
  }  
}  
  
layer {  
  name: "H1"  
  type: "InnerProduct"  
  bottom: "data"  
  top: "H1"  
  inner_product_param {  
    num_output: 2048     
  }  
}  
  
layer {  
  name: "H1_A"  
  type: "Sigmoid"  
  bottom: "H1"  
  top: "H1"  
}  
  
layer {  
  name: "H2"  
  type: "InnerProduct"  
  bottom: "H1"  
  top: "H2"  
  inner_product_param {  
    num_output: 2048  
  }  
}  
  
layer {  
  name: "H2_A"  
  type: "Sigmoid"  
  bottom: "H2"  
  top: "H2"  
}  
  
layer {  
  name: "H3"  
  type: "InnerProduct"  
  bottom: "H2"  
  top: "H3"  
  inner_product_param {  
    num_output: 2048  
  }  
}  
  
layer {  
  name: "H3_A"  
  type: "Sigmoid"  
  bottom: "H3"  
  top: "H3"  
}  
  
  
layer {  
  name: "L"  
  type: "InnerProduct"  
  bottom: "H3"  
  top: "L"  
  inner_product_param {  
    num_output: 26752   
  }  
}  
  
layer {  
  name: "loss"  
  type: "SoftmaxWithLoss"  
  bottom: "L"  
  bottom: "label"  
  top: "loss"  
} '''  %(batch_size)


output.write(out_str)
output.close()
output_solver.write(out_solver_str)
output_solver.close()
