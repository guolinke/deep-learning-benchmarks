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


local used_gpus = opt.gpus
used_gpus = stringx.split(used_gpus, ',')
for i = 1, #used_gpus do used_gpus[i] = tonumber(used_gpus[i]) + 1 end

function MakeFakeData(data_size, label_size, n_classes)
    return torch.randn(torch.LongStorage(data_size)), torch.IntTensor(label_size):random(1, n_classes)
end

function TimeRun(exec_fun, iter, info)
    nDryRuns = 10
    for i=1,nDryRuns do
        exec_fun()
    end
    cutorch.synchronize()
    collectgarbage()

    sys.tic()
    for i=1,iter do
        exec_fun()
    end
    cutorch.synchronize()
    collectgarbage()
    used_time = sys.toc()/iter

    print('Used time for ' ,info, ' : ',used_time)

    return used_time
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




local model, model_name, data_size, label_size, n_classes = GetModel(opt.batchSize / #used_gpus, used_gpus[1])

local inputCPU, labelCPU = MakeFakeData(data_size, label_size, n_classes)

local criterion


if #used_gpus > 1 then
    dp_model = nn.DataParallelTable(1)
    for i=1,#used_gpus do
        cutorch.setDevice(used_gpus[i])
        dp_model:add(model:clone():cuda(), used_gpus[i])
    end
    cutorch.setDevice(1)
else
    dp_model = model:cuda()
end

label = torch.IntTensor(label_size):cuda()
input = torch.Tensor(inputCPU:size()):float():cuda()
criterion = nn.ClassNLLCriterion():cuda()

local paramx, paramdx = dp_model:getParameters()
local ax, adx         = dp_model:parameters()
print('Model Parameters: ', paramx:nElement())
print('All shape: ', ax)


print('ModelType: ' .. model_name)


function RunFull()
    input:copy(inputCPU)
    label:copy(labelCPU)
    local preb = dp_model:forward(input)
    local err = criterion:forward(preb, label)
    dp_model:zeroGradParameters()
    dp_model:backward(input, criterion:backward(preb, label))
    dp_model:updateParameters(opt.LR)
    if dp_model.syncParameters then 
        dp_model:syncParameters() 
    end
end

batch_time = TimeRun(RunFull, opt.nIterations, '[copy + forward + backward + update]')

print(string.format("Avg elasped time per mini-batch (sec/mini-batch): %10.6f", (batch_time)))
print(string.format("Avg samples per second (samples/sec): %10.6f", opt.batchSize / batch_time))

