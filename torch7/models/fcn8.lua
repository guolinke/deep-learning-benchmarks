function fcn8(batch_size, deviceId)
    local input_size = 512 
    local hidden_size = 2048
    local output_size = 1000 
    require 'nn'
    -- Network definition
    local Linear = nn.Linear
    local Transfer = nn.Sigmoid
    local mlp = nn.Sequential()
    mlp:add(Linear(input_size,hidden_size)):add(Transfer(true)) -- hidden layer 1
    mlp:add(Linear(hidden_size,hidden_size)):add(Transfer(true)) -- hidden layer 2
    mlp:add(Linear(hidden_size,hidden_size)):add(Transfer(true)) -- hidden layer 3
    mlp:add(Linear(hidden_size,hidden_size)):add(Transfer(true)) -- hidden layer 4
    mlp:add(Linear(hidden_size,hidden_size)):add(Transfer(true)) -- hidden layer 5
    mlp:add(Linear(hidden_size,hidden_size)):add(Transfer(true)) -- hidden layer 6
    if deviceId >= 0 then
        mlp:add(Linear(hidden_size,output_size)):add(cudnn.LogSoftMax()) -- output layer
    else
        mlp:add(Linear(hidden_size,output_size)):add(nn.LogSoftMax()) -- output layer
    end

    return mlp, 'FCN8', {batch_size,input_size}, batch_size, output_size
end

return fcn8
