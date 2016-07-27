-- Usage th main.lua {train|test}

mode = arg[1]
assert (mode=='train' or mode=='test', "Bad arguments. Usage th main.lua {train|test}")

require 'cunn'
require 'fbcunn'
require 'cudnn'
require 'xlua'
require 'optim'
require 'math'
require 'gnuplot'
require 'sys'
require 'image'

mattorch = require('fb.mattorch');
pl = require'pl.import_into'()
debugger = require('fb.debugger');


paths.dofile('setting_options.lua');
cutorch.setDevice(config.GPU);
torch.manualSeed(config.GPU);
----------------------------

paths.dofile('utils.lua');
----------------------------
paths.dofile('data.lua');
----------------------------
paths.dofile('layers/RNNlayer.lua')
----------------------------------
paths.dofile('networks/ModelConstruction_IM.lua');
--------------------------------
paths.dofile('train_functions.lua');
------------------------------
log(config);


if mode == 'test' then	
	model:LoadModel(config.initModelPath.fullNN)
	log(model.fullNN)
	test()
else
	model:LoadModel("caffe");
	log(model.fullNN)
	train()
end

