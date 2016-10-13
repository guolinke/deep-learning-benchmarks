require 'sys';
require 'bit';
require 'optim';
require 'pl.stringx'

function initRequire()
    require 'nn'
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
    cudnn.verbose = false
    cudnn.fastest = true
end
initRequire()

local opts = require 'opts'
local opt = opts.parse(arg)


-- gpu transfer
local str = opt.gpus
str = stringx.split(str, ',')
for i = 1, #str do str[i] = tonumber(str[i]) end
opt.gpus = str
opt.deviceId = 1
print('opt.gpus = ', opt.gpus)
print('opt.deviceId = ', opt.deviceId)

for i = 1, #opt.gpus do opt.gpus[i] = opt.gpus[i] + 1 end
opt.threads = #opt.gpus

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

function MakeFakeData(data_size, label_size, n_classes, n_threads)
    assert(data_size[1] % n_threads == 0, 'please make sure batch_size % n_threads == 0')
    data_size[1] = math.floor(data_size[1] / n_threads)
    label_size   = math.floor(label_size / n_threads)
    return torch.randn(torch.LongStorage(data_size)), torch.IntTensor(label_size):random(1, n_classes)
end


initRequire()
cutorch.setDevice(opt.gpus[1]) -- set as main gpu
print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

collectgarbage()

local model, model_name, data_size, label_size, n_classes = GetModel(opt.batchSize, opt.deviceId)
local criterion = nn.ClassNLLCriterion()
local inputCPU, labelCPU = MakeFakeData(data_size, label_size, n_classes, opt.threads)

--local paramx, paramdx       = model:getParameters()
local ap, dap = model:parameters()
local global_weights, adx   = model:getParameters()
local tmp_weights   = global_weights:clone()
--local tmp_weights = {}
--for i = 1, #global_weights do table.insert(tmp_weights, global_weights[i]:clone()) end
print('Model Parameters: ', global_weights:nElement())
print('All shape: ', ap)
print('ModelType: ' .. model_name)
--local tds = require 'tds'

local local_models     = {} --tds.Vec()
local local_weights    = {} --tds.Vec()
local local_dweights   = {} --tds.Vec()
local local_criterions = {} --tds.Vec()
local local_inputs     = {} --tds.Vec()
local local_labels     = {} --tds.Vec()
for t = 1, opt.threads do
    cutorch.setDevice(opt.gpus[t])
    --local_models[t] = model:clone('weight', 'bias'):float():cuda()
    table.insert(local_models, model:clone('weight', 'bias'):float():cuda())
    local weights, dweights = local_models[#local_models]:getParameters()
    --local_weights[t] = weights
    --local_dweights[t] = dweights
    --local_criterions[t] = criterion:clone():float():cuda()
    --local_inputs[t]  = torch.Tensor(inputCPU:size()):float():cuda()
    --local_labels[t]  = torch.IntTensor(labelCPU:size()):float():cuda()
    table.insert(local_weights, weights)
    table.insert(local_dweights, dweights)
    table.insert(local_criterions, criterion:clone():float():cuda())
    table.insert(local_inputs, torch.Tensor(inputCPU:size()):float():cuda())
    table.insert(local_labels, torch.IntTensor(labelCPU:size()):float():cuda())
end


function RunFull()
    --sys.tic()
    for t = 1, opt.threads do
        cutorch.setDevice(opt.gpus[t])
        local_inputs[t]:copy(inputCPU)
        local_labels[t]:copy(labelCPU)

        local input  = local_inputs[t]
        local label  = local_labels[t]
        local module = local_models[t]
        local criterion = local_criterions[t]

        local z = module:forward(input)
        local err = criterion:forward(z, label)
        module:zeroGradParameters()
        module:updateGradInput(input, criterion:updateGradInput(module.output, label))
        module:accGradParameters(input, criterion.gradInput)
    end
    cutorch.synchronize()
    --print('forward backward time = ', sys.toc()) --sys.tic()

    --sys.tic()
    cutorch.setDevice(opt.gpus[1]) -- set as main gpu
    for t = 1, opt.threads do
        tmp_weights:copy(local_dweights[t])
        global_weights:add(-opt.LR, tmp_weights)
    end
    cutorch.synchronize()
    --print('global_weights acc time = ', sys.toc()) --sys.tic()

    --sys.tic()
    -- spread
    for t = 1, opt.threads do
        cutorch.setDevice(opt.gpus[t])
        local_weights[t]:copy(global_weights)
    end
    cutorch.synchronize()
    --print('spread time = ', sys.toc()) --sys.tic()
end


function TimeRun(exec_fun, iter, info)
    print('try RUN begin')
    nDryRuns = 10
    for i=1,nDryRuns do
        exec_fun()
    end
    cutorch.synchronize()
    collectgarbage()

    print('test RUN begin')
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

batch_time = TimeRun(RunFull, opt.nIterations, '[copy + forward + backward + update]')

print(string.format("Avg elasped time per mini-batch (sec/mini-batch): %10.6f", (batch_time)))
print(string.format("Avg samples per second (samples/sec): %10.6f", opt.batchSize / batch_time))

