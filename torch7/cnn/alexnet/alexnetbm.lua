require 'sys'
require 'optim'
require 'cutorch'


local opts = require 'opts'
local opt = opts.parse(arg)

-- require 'fbcunn'
-- require 'nnbhwd' -- not compiling anymore, file an issue
local nets = {}
nets[#nets+1] = require 'alexnet'
-- nets[#nets+1] = require 'overfeat'
-- nets[#nets+1] = require 'vgg_a'
--nets[#nets+1] = require 'googlenet'

local libs = {}
-- libs[#libs+1] = {fbnn.SpatialConvolution, cudnn.SpatialMaxPooling, cudnn.ReLU, 'BDHW', 'fbnn'}
-- libs[#libs+1] = {nn.SpatialConvolutionMM, nn.SpatialMaxPooling, nn.ReLU, 'BDHW', 'nn'}
-- libs[#libs+1] = {nn.SpatialConvolutionBHWD, nn.SpatialMaxPoolingBHWD, nn.ReLU, 'BHWD', 'nnBHWD'}

if opt.deviceId >= 0 then
   require 'cunn'
   require 'cudnn'
   cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
   cudnn.verbose = false
   cutorch.setDevice(opt.deviceId+1)
   print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)
   libs[#libs+1] = {cudnn.SpatialConvolution, cudnn.SpatialMaxPooling, cudnn.ReLU, 'BDHW', 'cudnn'}
else
   require 'nn'
   libs[#libs+1] = {nn.SpatialConvolution, nn.SpatialMaxPooling, nn.ReLU, 'BDHW', 'nn'}
end

steps = opt.nIterations-- nb of steps in loop to average perf
nDryRuns = 20

function makeInput(config, size)
   local layout = config[4]
   local osize, lsize
   if layout == 'BDHW' then
      osize = size
   elseif layout == 'DHWB' then
      osize = {size[2],size[3],size[4],size[1]}
   elseif layout == 'BHWD' then
      osize = {size[1], size[3], size[4], size[2]}
   end
   lsize = size[1]
   return torch.randn(torch.LongStorage(osize))
end

for i=1,#nets do
   for j=1,#libs do
      collectgarbage()
      local model,model_name,size = nets[i](libs[j])
      local paramx, paramdx = model:getParameters()
      local ax, adx         = model:parameters()
      print('Model Parameters: ', paramx:nElement())
      print('All shape: ', ax)

      size[1] = opt.batchSize
      local inputCPU, labelCPU
      local criterion

      if opt.deviceId >= 0 then
          model = model:cuda()
          inputCPU = makeInput(libs[j],size)
          labelCPU = torch.IntTensor(opt.batchSize):random(1, 1000)
          label = torch.IntTensor(opt.batchSize):cuda()
          input = torch.Tensor(inputCPU:size()):float():cuda()
          criterion = nn.ClassNLLCriterion():cuda()
      else
          inputCPU = makeInput(libs[j],size)
          label = torch.IntTensor(opt.batchSize):random(1, 1000)
          input = torch.Tensor(inputCPU:size()):float()
          criterion = nn.ClassNLLCriterion()
      end
      local lib_name = libs[j][5]
      print('ModelType: ' .. model_name, 'Kernels: ' .. lib_name,
            'Input shape: ' .. input:size(1) .. 'x' .. input:size(2) ..
               'x' .. input:size(3) .. 'x' .. input:size(4))

      -- dry-run
      for i=1,nDryRuns do
         input:copy(inputCPU)
         label:copy(labelCPU)
         local preb = model:forward(input)
         criterion:forward(preb, label)
         model:zeroGradParameters()
         model:backward(input, criterion:backward(preb, label))
         model:updateParameters(opt.LR)
      end

        
      collectgarbage()
      sys.tic()
      for t = 1, steps do
         input:copy(inputCPU)
         label:copy(labelCPU)
         local preb = model:forward(input)
         criterion:forward(preb, label)
         model:zeroGradParameters()
         model:backward(input, criterion:backward(preb, label))
         model:updateParameters(opt.LR)
      end
      cutorch.synchronize()

      total_time = sys.toc()/steps
      print(string.format(" | Epoch: [][]    Time %10.6f", (total_time)))
      print()
   end
end

print('')
