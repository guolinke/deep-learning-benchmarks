require 'sys';
require 'bit';
require 'optim';
require 'cutorch'

local opts = require 'opts'
local opt = opts.parse(arg)

local GetModel 

if opt.arch == 'alexnet' then
    GetModel = require 'models/alexnet'
elseif opt.arch == 'resnet' then
    GetModel = require 'models/resnet'
elseif opt.arch == 'fcn5' then
    GetModel = require 'models/fcn5'
elseif opt.arch == 'fcn8' then
    GetModel = require 'models/fcn8'
end

function MakeFakeData(data_size, label_size, n_classes)
    return torch.randn(torch.LongStorage(data_size)), torch.IntTensor(label_size):random(1, n_classes)
end

if opt.deviceId >= 0 then
    require 'cunn'
    require 'cudnn'
    cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
    cudnn.verbose = false
    cudnn.fastest = true
    cutorch.setDevice(opt.deviceId+1)
    print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)
end

collectgarbage()

local model, model_name, data_size, label_size, n_classes = GetModel(opt.batchSize, opt.deviceId)

local paramx, paramdx = model:getParameters()
local ax, adx         = model:parameters()
print('Model Parameters: ', paramx:nElement())
print('All shape: ', ax)

local inputCPU, labelCPU = MakeFakeData(data_size, label_size, n_classes)

local criterion

if opt.deviceId >= 0 then
    model = model:cuda()
    label = torch.IntTensor(label_size):cuda()
    input = torch.Tensor(inputCPU:size()):float():cuda()
    criterion = nn.ClassNLLCriterion():cuda()
else
    input = torch.Tensor(inputCPU:size()):float()
    label = torch.IntTensor(label_size)
    criterion = nn.ClassNLLCriterion()
end
print('ModelType: ' .. model_name)

-- dry-run
nDryRuns = 20
for i=1,nDryRuns do
    input:copy(inputCPU)
    label:copy(labelCPU)
    local preb = model:forward(input)
    criterion:forward(preb, label)
    model:zeroGradParameters()
    model:backward(input, criterion:backward(preb, label))
    model:updateParameters(opt.LR)
end
cutorch.synchronize()
collectgarbage()

sys.tic()
for t = 1, opt.nIterations do
    input:copy(inputCPU)
    label:copy(labelCPU)
    local preb = model:forward(input)
    criterion:forward(preb, label)
    model:zeroGradParameters()
    model:backward(input, criterion:backward(preb, label))
    model:updateParameters(opt.LR)
end
cutorch.synchronize()
batch_time = sys.toc()/opt.nIterations

print(string.format("Avg elasped time per mini-batch (sec/mini-batch): %10.6f", (batch_time)))
print(string.format("Avg samples per second (samples/sec): %10.6f", opt.batchSize / batch_time))
print()
