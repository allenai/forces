--Constructing The NN model
log('Constructing Network Model ..... \n');
---------------------------------------
cudnn = require('cudnn')

model={};
model.imageNN = require('models.alexnetOrig')
model.jointNN = require('models.rnn_unfold')

model.criterion = nn.ParallelCriterion()
for i = 1,config.rho do
  model.criterion:add(nn.CrossEntropyCriterion(config.cweight))
end
model.criterion:cuda()

function model:LearningRateComp(iter)
  local lIter = (iter % config.nResetLR)+1;
  local regimes= config.regimes;
  for _, row in ipairs(regimes) do
    if lIter >= row[1] and lIter <= row[2] then
      return row[3];
    end
  end
end

function model:TrainOneBatch(input,target)
  -- Set into training phase (just activates the droputs)
  model.fullNN:training();

  
  -- Forward passs
  model.fullNN:forward(input);
  
  local loss = model.criterion:forward(model.fullNN.output,target)
  local acc = {}
  local per_class = {}

  local predictedLabel_all = {}
  
  for k = 1,config.rho do
     local max,predictedLabel = torch.max(model.fullNN.output[k],2);
     predictedLabel = predictedLabel[{{}, 1}]
     table.insert(predictedLabel_all, predictedLabel)
     acc[k], per_class[k] = GetPerClassAccuracy(predictedLabel, target[k])
    
  end
  
  local acc_all = GetPerClassAccuracyRNN(predictedLabel_all, target, config.rho)
  
  -- Make sure gradients are zero
  model.fullNN:zeroGradParameters();

  -- Backward pass
  local endIndicator = torch.zeros(1,config.nCategories);
  endIndicator[1][config.nCategories] = 1;


  for i= 1, config.rho do
    local end_mask = target[i]:eq(config.nCategories):float()
    local end_idx = end_mask:nonzero()
    if end_mask:sum() >0 then 
       for j = i+1 , config.rho do 
           endP = endIndicator:clone():repeatTensor(end_idx:size(1),1);
           model.fullNN.output[j]:indexCopy(1,end_idx,endP:cuda())
       end
    end
  end


  local bwCri = model.criterion:backward(model.fullNN.output,target)
  model.fullNN:backward(input,bwCri);
  
  -- updating the weights
  model.fullNN:updateParameters(model.learningRate);
  return  acc,per_class,loss, acc_all
  
end


function model:EvaluateOneBatch(input,target)
  -- Set into Evaluation mode (just deactivates the dropouts)
  model.fullNN:evaluate();
  -- Forward passs
  model.fullNN:forward(input);
  local predictedLabel_all = {}

  local loss = model.criterion:forward(model.fullNN.output,target)
  local acc = {}
  local per_class = {}
  for k = 1,config.rho do
     local max, predictedLabel = torch.max(model.fullNN.output[k],2);
     predictedLabel = predictedLabel[{{}, 1}]
     table.insert(predictedLabel_all, predictedLabel)
     acc[k], per_class[k] = GetPerClassAccuracy(predictedLabel, target[k])
  end
  local acc_all = GetPerClassAccuracyRNN(predictedLabel_all, target, config.rho)

  return acc, per_class, loss, acc_all, predictedLabel_all
end

function model:LoadCaffeImageNN(caffeFilePath)
  local protoFile = caffeFilePath.proto
  local modelFile = caffeFilePath.model
  local meanFile  = caffeFilePath.mean

  require 'loadcaffe'
  local caffeModel = loadcaffe.load(protoFile,modelFile,'cudnn');
  
  caffeModel:remove(24);
  caffeModel:remove(23);
  caffeModel:remove(22);
  local caffeParams = GetNNParamsToCPU(caffeModel);

  local nChn = GetValuesSum(GetEnableInputTypes(config.input_data));
  
  nChn = nChn / 3 -- we assume each additional feature has 3 channels
  caffeParams[1] = caffeParams[1]:repeatTensor(1, nChn, 1, 1)

  
  LoadNNlParams(model.imageNN, caffeParams);
  model.jointNN:apply(rand_initialize)
  LoadCaffeMeanStd(meanFile);
end


function model:SaveModel(fileName)
  local saveModel ={};
  -- reading model parameters to CPU
  saveModel.imageNN     = GetNNParamsToCPU(model.imageNN);
  saveModel.jointNN     = GetNNParamsToCPU(model.jointNN);
  -- saving into the file
  torch.save(fileName,saveModel)
end

function model:LoadModel(fileName)
  if fileName=="caffe" then
     if fileName ~= "" then
      log('WARNING: specified initialized model is ignored because caffe model will be loaded!!!!!!!!!!!')
     end
     log('Loading Network Model from caffe')
     model:LoadCaffeImageNN(config.caffeFilePath);

  else
    if fileName ~= "" then
      log('Loading Network Model from ' .. fileName)
      local saveModel = torch.load(fileName);
      LoadNNlParams(model.imageNN ,saveModel.imageNN);
      LoadNNlParams(model.jointNN ,saveModel.jointNN);
      if config.caffeInit then 
        LoadCaffeMeanStd(config.caffeFilePath.mean);
      end      
    else
      -- Initialize the model randomly
      model.imageNN:apply(rand_initialize);
      model.jointNN:apply(rand_initialize);
    end
  end

  model.IF_NN = nn.ParallelTable():add(model.imageNN):add(nn.Identity())
  model.fullNN = nn.Sequential():add(model.IF_NN):add(nn.JoinTable(2)):add(model.jointNN)
  model.fullNN:cuda();

  model:SaveModel(config.logDirectory .. '/init.t7')
end
