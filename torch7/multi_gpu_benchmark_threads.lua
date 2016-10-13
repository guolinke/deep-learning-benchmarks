require 'sys';
require 'bit';
require 'optim';
--require 'cutorch'
require 'pl.stringx'
function initRequire()
    print('begin init')
    require 'nn'
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
    cudnn.verbose = false
    cudnn.fastest = true
    print('end init')
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

local paramx, paramdx       = model:getParameters()
local global_weights, adx   = model:parameters()
local tmp_weights = {}
for i = 1, #global_weights do table.insert(tmp_weights, global_weights[i]:clone()) end
print('Model Parameters: ', paramx:nElement())
print('All shape: ', global_weights)
print('ModelType: ' .. model_name)



local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
print('init threads')
local threads = Threads(
opt.threads,
function() 
    require 'nn' require 'cunn' require 'cudnn' require 'cutorch' 
    cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
    cudnn.verbose = false
    cudnn.fastest = true
end,
function(threadid) 
    print('set gpu id = ', threadid)
    cutorch.setDevice(opt.gpus[threadid])

    local module = model:clone('weight', 'bias'):float():cuda()
    local weights, dweights = module:parameters()
    --table.insert(local_weights, threadid, weights)
    --local_weights[threadid] = weights
    --print('local_weights['..threadid..']= ' .. local_weights[threadid])
    local criterion = criterion:clone():float():cuda()
    local input     = torch.Tensor(inputCPU:size()):float():cuda()
    local label     = torch.IntTensor(labelCPU:size()):float():cuda()
    function gupdate() --inputCPU, labelCPU)
        input:copy(inputCPU)
        label:copy(labelCPU)
        local z = module:forward(input)
        local err = criterion:forward(z, label)
        module:zeroGradParameters()
        module:updateGradInput(input, criterion:updateGradInput(module.output, label))
        module:accGradParameters(input, criterion.gradInput)
        return dweights
    end
    function copyback()
        cutorch.setDevice(opt.gpus[__threadid])
        for i = 1, #weights do
            weights[i]:copy(global_weights[i])
        end
    end
end
)


function RunFull()
    for idx = 1, opt.threads do
        threads:addjob(
        function(idx) return idx, gupdate() end,
            --inputCPU, labelCPU) end,
        function(idx, dweights)
            --print('update begin in thread ', idx)
            cutorch.setDevice(opt.gpus[1]) -- acc in main gpu

            for i = 1, #global_weights do
                tmp_weights[i]:copy(dweights[i])
                global_weights[i]:add(-opt.LR, tmp_weights[i])
            end
        end,
        idx
        )
    end
    threads:synchronize()
    cutorch.synchronize()

    -- print('spread out')
    -- spread out
    for idx = 1, opt.threads do
        threads:addjob(
        function(idx)  
            copyback()
            --print('get global_weights done thread = ' .. __threadid)
        end,
        function() end,
        idx
        )
    end
    threads:synchronize()
    cutorch.synchronize()
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
threads:terminate()

print(string.format("Avg elasped time per mini-batch (sec/mini-batch): %10.6f", (batch_time)))
print(string.format("Avg samples per second (samples/sec): %10.6f", opt.batchSize / batch_time))

