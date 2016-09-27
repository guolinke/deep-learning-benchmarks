function fcn5(batch_size, deviceId)
    local input_size = 26752 
    local hidden_size = 2048
    local output_size = 26752 
    require 'nn'
    local Linear = nn.Linear
    local Transfer = nn.Sigmoid
    -- Network definition
    local mlp = nn.Sequential()
    mlp:add(Linear(input_size,hidden_size)):add(Transfer(true)) -- hidden layer 1
    mlp:add(Linear(hidden_size,hidden_size)):add(Transfer(true)) -- hidden layer 2
    mlp:add(Linear(hidden_size,hidden_size)):add(Transfer(true)) -- hidden layer 3
    if deviceId >= 0 then
        mlp:add(Linear(hidden_size,output_size)):add(cudnn.LogSoftMax()) -- output layer
    else
        mlp:add(Linear(hidden_size,output_size)):add(nn.LogSoftMax()) -- output layer
    end

    return mlp, 'FCN5', {batch_size,input_size}, batch_size, output_size
end

return fcn5

