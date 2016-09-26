--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)
local bsize = 16
local tsize = bsize * 2 
local osize = 1000
local inputCPU = nil -- torch.randn(torch.LongStorage({bsize,3,224,224})):type('torch.FloatTensor')
local targetCPU = nil 

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   bsize = self.opt.batchSize
   inputCPU = torch.randn(torch.LongStorage({bsize,3,224,224})):type('torch.FloatTensor')
   targetCPU = torch.IntTensor(bsize):random(1,osize)
end

function Trainer:train(epoch, dataloader, iter)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)
   tsize = bsize * iter

   

   local function feval()
      return self.criterion.output, self.gradParams
   end

   nDryUp = 20
   for i = 1, nDryUp  do
      -- Copy input and target to the GPU
      self:copyInputs()


      local output = self.model:forward(self.input):float()
      self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

   end

   print('=> Training epoch # ' .. epoch)

   local timer = torch.Timer()
   -- set the batch norm to training mode
   self.model:training()
   for i = 1, iter  do
      -- Copy input and target to the GPU
      self:copyInputs()

      local output = self.model:forward(self.input):float()
      self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

   end

   print((' | Epoch: [][]    Time   %.3f '):format(
         timer:time().real / iter))
   return 0,0,0
end


function Trainer:copyInputs()
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   local bsize = self.opt.batchSize
   -- self.input = self.input or (self.opt.nGPU == 1
   --    and torch.CudaTensor()
   --    or cutorch.createCudaHostTensor())
   -- self.target = self.target or torch.CudaTensor()

   -- self.input:resize(sample.input:size()):copy(sample.input)
   -- self.target:resize(sample.target:size()):copy(sample.target)
   if self.opt.deviceId < 0 then
       print 'not use cuda'
       self.input = inputCPU 
       self.target = targetCPU -- torch.IntTensor(bsize):random(1,osize)
   else
       self.input = inputCPU:cuda() -- torch.CudaTensor(inputCPU:size())
       self.target = targetCPU:cuda() -- torch.IntTensor(bsize):random(1,osize):cuda()
   end
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
