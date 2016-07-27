---- options

config={};

config.GPU = 1

config.DataRootPath = "/media/drive4/force_data"
config.SaveRootPath = "/media/drive4/force_data/logs"
config.CacheRootPath = config.DataRootPath

config.logDirectory = config.SaveRootPath .. '/' .. "LOG_" .. os.getenv('USER') .. "_" .. os.date():gsub(' ','-');
os.execute('mkdir -p ' .. config.logDirectory)
config.logFile = assert(io.open(paths.concat(config.logDirectory, 'log.txt'), 'w'))
config.imgFilenamesLog = assert(io.open(paths.concat(config.logDirectory, 'img_filenames.txt'), 'w'))

config.imH = 227;
config.imW = 227;


config.input_data = {
  annotation = {
    dir = config.DataRootPath .. "/annotations",
  },
  image = {
    dir       = config.DataRootPath .. "/rgbimages",
    nChannels = 3,
    type      = "png",
    suffix    = "im",
    mean      = {},
    std       = {},
    enable    = true,
    croppable = true,
  },
  force = {
    dir       = config.DataRootPath .. "/../alexnet_force/alexnet_savedir_2d",
    nChannels = 0,
    type      = "png",
    suffix    = "forces",
    mean      = {},
    std       = {},
    enable    = true,
  },
  mask = {
    dir       = config.DataRootPath .. "/objmasks",
    nChannels = 3,
    type      = "png",
    suffix    = "mask",
    mean      = {},
    std       = {},
    enable    = true,
  },  
  depth = {
    dir       = config.DataRootPath .. "/depths",
    nChannels = 3,
    type      = "png",
    suffix    = "depth",
    mean      = {},
    std       = {},
    enable    = false,
  },    

}

trainmeta = {
  save_dir = config.CacheRootPath .. "/train_savedir",
  datafile = config.DataRootPath .. "/data_rotation/data_trainval.txt",
}

trainvalmeta = {
  save_dir = config.CacheRootPath .. "/trainval_savedir",
  datafile = config.DataRootPath .. "/data_trainval.txt",
  fileids  = config.DataRootPath .. "/trainvalIDs.txt",
}

valmeta = {
  save_dir = config.CacheRootPath .. "/val_savedir",   
  datafile = config.DataRootPath .. "/data_val.txt",
  fileids  = config.DataRootPath .. "/valIDs.txt",
}

testmeta = {
  save_dir = config.CacheRootPath .. "/test_savedir",
  datafile = config.DataRootPath .. "/data_rotation/data_test.txt",
  outfile  = config.logDirectory .. "/predictions.mat",
}

config.train = trainmeta
config.trainval = trainvalmeta
config.val = valmeta
config.test = testmeta


--------   BEGIN: Network configuration  -----
config.nIter    = 15010 -- for test: 419
config.nDisplay = 1;
config.batchSize = 128 -- for test: 83
config.saveModelIter = 1000;
config.nResetLR = 50000;
config.nEval    = 10;
config.lambda   = 0.5

config.imageNetType     = 'alexnet';

config.caffeInit = true;
config.caffeFilePath = {  
  proto  = config.DataRootPath .. '/caffeModelZoo/deploy.prototxt',
  model  = config.DataRootPath .. '/caffeModelZoo/bvlc_alexnet.caffemodel',
  mean   = config.DataRootPath .. '/caffeModelZoo/ilsvrc_2012_mean.mat'
};
config.initModelPath = {
        fullNN = "./Model_iter_15000.t7" }
              



config.regimes = {
    -- start, end,    LR,
    {  1,     1000,   1e-3, },
    { 1001,     5000,   1e-4, },
    { 5001,     10000,   1e-3, },
    {10001,      15010,  1e-4,},
};

config.dropoutProb = 0.5;
config.cur_pointer = 1;
config.rho = 6;
config.nPredDim = config.rho ; 
config.RNNhSize = 1000;
config.RNNoSize = 18;
config.nCategories = 18;


-- These weights are based on the frequency of each direction in training data
config.cweight = torch.Tensor({0.0039, 0.0027, 0.0026, 0.0027, 0.0037, 0.0125,
                               0.0245, 0.0135, 0.0493, 0.0410, 0.0335, 0.0366,
                               0.0480, 0.1305, 0.3688, 0.1687, 0.0565, 0.0028})

config.force_size = 4096;
--------   END :  Network configuration -------

