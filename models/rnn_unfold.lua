
local start = nn.ConcatTable()
                    :add(nn.Identity())
                    :add(nn.Sequential():add(nn.Linear(4096+config.force_size,config.RNNhSize)):add(nn.ReLU(true)))
return nn.Sequential():add(start):add(rnn(4096+config.force_size, config.RNNhSize, config.RNNoSize, config.rho, false))

