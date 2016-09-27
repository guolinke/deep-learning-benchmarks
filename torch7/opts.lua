--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 Training script')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-arch',    'alexnet', 'Options: resnet | preresnet')
   cmd:option('-deviceId',       1,          'Specify GPU id')
   cmd:option('-nIterations',  100,          'Number of iterations for each epoch')
   cmd:option('-batchSize',       16,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-LR',              0.01,   'initial learning rate')

   cmd:text()

   local opt = cmd:parse(arg or {})

   return opt
end

return M
