function RemoveDotDirs(aTable)
  if aTable == nil or type(aTable) ~= 'table' then
    return aTable
  end
  --remove the first two directories "." , ".."
  local i = 1
  while i <= #aTable do
    while aTable[i] ~= nil and aTable[i]:sub(1,1) == '.' do
      aTable[i] = aTable[#aTable]
      aTable[#aTable] = nil
    end
    i = i + 1
  end
end

function getTableSize(aTable)
  local numItems = 0
  for k,v in pairs(aTable) do
      numItems = numItems + 1
  end
  return numItems
end

function GetRandomValue(aTable)
  local values = {}
  for key, value in pairs(aTable) do
    values[ #values+1 ] = value
  end
  return values[ torch.random(#values) ]
end

function GetValuesSum(aTable)
  local total = 0
  for key, value in pairs(aTable) do
    total = total + value
  end
  return total
end

function loadImageOrig(path)
  -----------------------------------------------------------------
  -- Reads an image
  -- inputs:
  --        "path": path to the image
  -- output:
  --        "im": the image
  -----------------------------------------------------------------

  local im = image.load(path, 3, "byte"):float()
      if im:dim() == 2 then -- 1-channel image loaded as 2D tensor
      im = im:view(1,im:size(1), im:size(2)):repeatTensor(3,1,1)
   elseif im:dim() == 3 and im:size(1) == 1 then -- 1-channel image
      im = im:repeatTensor(3,1,1)
   elseif im:dim() == 3 and im:size(1) == 3 then -- 3-channel image
   elseif im:dim() == 3 and im:size(1) == 4 then -- image with alpha
      im = im[{{1,3},{},{}}]
   else
      error("image structure not compatible")
   end
   return im
end

function loadImage(path, imH, imW)
  -----------------------------------------------------------------
  -- Reads an image and rescales it
  -- inputs:
  --        "path": path to the image
  --        "imH" and "imW": the image is rescaled to imH x imW
  -- output:
  --        "im": the rescaled image
  -----------------------------------------------------------------
   local im = loadImageOrig(path)
   --debugger.enter()
   im = image.scale(im, imW, imH)
   local tmp = im[{{1},{},{}}]
   im[{{1},{},{}}] = im[{{3},{},{}}]
   im[{{3},{},{}}] = tmp

   return im
end

function normalizeImage(im, mean, std)
  -----------------------------------------------------------------
  -- Normalizes image "im" by subtracting the "mean" and dividing by "std"
  -----------------------------------------------------------------
  for channel=1,3 do
    im[{channel,{},{}}]:add(-mean[channel]);
    im[{channel,{},{}}]:div(std[channel]);
  end
  return im;
end

function LoadRandomSamples(nSamples, allfiles, imH, imW)
  -----------------------------------------------------------------
  -- Loads "nSamples" images from the "allfiles" and rescaled them to imH x imW
  -- inputs:
  --       nSamples: # of images that is sampled
  --       allfiles: an array of paths of the images in the dataset
  --       imH, imW: size of the rescaled image
  -- outputs:
  --       images: 4D Tensor that includes "nSamples" number of imHximW images
  -----------------------------------------------------------------
  nIms = math.min(nSamples, #allfiles)
  local images = torch.Tensor(nIms, 3, imH, imW);
  local randnums = torch.randperm(nIms);
  local idx = randnums[{{1,nIms}}];
  for i = 1,nIms do
    local fname = allfiles[idx[i]];
    local im = loadImage(fname, imH, imW);
    images[{{i},{},{},{}}] = im;
  end
  return images;
end

function ComputeMeanStd(nSample, allfiles, imH, imW)
  -----------------------------------------------------------------
  -- Computes the mean and std of randomly sampled images
  -- inputs:
  --       nSample: # of images that is sampled
  --       allfiles: an array of paths of the images in the dataset
  --       imH, imW: size of the rescaled image
  -- outputs:
  --       mean: a 3-element array (the mean for each channel)
  --       std:  a 3-element array (the std for each channel)
  -----------------------------------------------------------------

  local images    = LoadRandomSamples(nSample, allfiles, imH, imW);
  local mean = {};
  local std  = {};

  mean[1]   = torch.mean(images[{{},1,{},{}}]);
  mean[2]   = torch.mean(images[{{},2,{},{}}]);
  mean[3]   = torch.mean(images[{{},3,{},{}}]);

  std[1]    = torch.std(images[{{},1,{},{}}]);
  std[2]    = torch.std(images[{{},2,{},{}}]);
  std[3]    = torch.std(images[{{},3,{},{}}]);

  return mean, std;
end

function MakeListTrainFrames(dataset, trainDir, image_type)
  allfiles = {};
  for category, subdataset in pairs(dataset) do
    -- TODO(hessam): Resolve the hacky solution
    if category ~= 'config' then
      for angles, subsubdataset in pairs(subdataset) do
        for dirs, files in pairs(subsubdataset) do
          for _, f in pairs(files) do
            fname = string.sub(f, 1, -11) .. "." .. image_type;
            table.insert(allfiles, paths.concat(trainDir, category, dirs, fname));
          end
        end
      end
    end
  end
  return allfiles;
end

function MakeListGEFrames(dataset, data_type)
  local geDir   = config.GE.dir;
  allfiles = {};
  for categories, subdataset in pairs(dataset) do
    for angles, subsubdataset in pairs(subdataset) do
      for dirs, files in pairs(subsubdataset) do
        for _, f in pairs(files) do
          table.insert(allfiles, paths.concat(geDir, categories, categories .. "_" .. angles .. "_" .. data_type, dirs, f));
        end
      end
    end
  end
  return allfiles;
end

function shuffleList(list, deterministic)
  local rand
  if deterministic then -- shuffle! but deterministicly.
    math.randomseed(2)
    rand = math.random
  else
    rand = torch.random
  end

  for i = #list, 2, -1 do
      local j = rand(i)
      list[i], list[j] = list[j], list[i]
  end
end


function MakeShuffledTuples(dataset, deterministic)
  local trainDir   = config.trainDir;
  tuples = {};
  for category, subdataset in pairs(dataset) do    
    if category ~= 'config' then
      local physicsCategory = GetPhysicsCategory(category)
      for angles, subsubdataset in pairs(subdataset) do
        for dirs, _ in pairs(subsubdataset) do
          table.insert(tuples, {category, physicsCategory, angles, dirs});
        end
      end
    end
  end
  shuffleList(tuples, deterministic);
  return tuples;
end

function isExcluded(excluded_categories, category)
  for _, ecat in pairs(excluded_categories) do
    if category:find(ecat) then
      return true
    end
  end
  return false
end


function GetNNParamsToCPU(nnModel)
  -- Convert model into FloatTensor and save.
   local params, gradParams = nnModel:parameters()
   if params ~= nill then
      paramsCPU = pl.tablex.map(function(param) return param:float() end, params)
    else
      paramsCPU = {};
    end
    return paramsCPU
end

function LoadNNlParams(current_model,saved_params)
    local params, gradparams = current_model:parameters()
    if params ~= nill then
      assert(#params == #saved_params,
        string.format('#layer != #saved_layers (%d vs %d)!',
          #params, #saved_params));
      for i = 1,#params do
        assert(params[i]:nDimension() == saved_params[i]:nDimension(),
          string.format("Layer %d: dimension mismatch (%d vs %d).",
            i, params[i]:nDimension(), saved_params[i]:nDimension()))
        for j = 1, params[i]:nDimension() do
          assert(params[i]:size(j) == saved_params[i]:size(j),
            string.format("Layer %d, Dim %d: size does not match (%d vs %d).",
              i, j, params[i]:size(j), saved_params[i]:size(j)))
        end
        params[i]:copy(saved_params[i]);
      end
    end
end

function rand_initialize(layer)

  local tn = torch.type(layer)
  if tn == "cudnn.SpatialConvolution" then
    local c  = math.sqrt(10.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "cudnn.VolumetricConvolution" then
    local c  = math.sqrt(10.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.Linear" then
    local c =  math.sqrt(10.0 / layer.weight:size(2));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    --layer.weight:fill(0)
    layer.bias:fill(0)
--  elseif layer.weight or layer.bias then
--    -- If there is any parameterized layer that skips the new initialization,
--    -- fail early so that users could notice.
--    error("Layer is trainable but not initialized!");
  end
end



function GetUniformRandomElement(data)
  local result = {}
  while type(data) == 'table' do
    local keys = {}
    for key, value in pairs(data) do
      if key ~= 'config' and (type(value) ~= 'table' or next(value) ~= nil) then
        keys[ #keys+1 ] = key
      end
    end
    local randomKey = keys[torch.random(#keys)]
    data = data[randomKey]
    result[#result+1] = randomKey
  end
  result[#result+1] = data
  return result
end


function log(...)
  -- Log to file:
  io.output(config.logFile)
  print(...)
  -- Log to stdout:
  io.output(io.stdout)
  print(...)
end





function Choose(tensor, indices)
  assert(tensor:size(1) == indices:size(1), "Dimension mismatch")
  local result = torch.Tensor( indices:size() )
  for i = 1, indices:size(1) do
    result[i] = tensor[i][ indices[i] ]
  end
  return result:cuda()
end

function ContainsValue(dict, value)
  for k,v in pairs(dict) do
    if v == value then
      return true
    end
  end
  return false
end

function GetPerClassAccuracy(predictions, labels)
  local per_class = torch.Tensor(config.nCategories, 2):fill(0)
  local nAccurate = 0
  labels = torch.squeeze(labels:clone())
  predictions = predictions:clone()

  for i=1,labels:size(1) do
    if labels[i] == predictions[i] then
      nAccurate = nAccurate + 1
      per_class[ labels[i] ][1] = per_class[ labels[i] ][1] + 1
    end
    per_class[ labels[i] ][2] = per_class[ labels[i] ][2] + 1
  end

  local acc = nAccurate / labels:size(1)
  
  local nonzero_cls = torch.nonzero(per_class[{{},{2}}])
  nonzero_cls = torch.squeeze(nonzero_cls[{{},{1}}])
  --debugger.enter()
  if torch.isTensor(nonzero_cls) == false then
    nonzero_cls = torch.LongTensor(1):fill(nonzero_cls)
  end
  per_class = per_class:index(1,nonzero_cls)

  per_class = torch.mean(torch.cdiv(per_class[{{},{1}}], per_class[{{},{2}}]))
  return acc, per_class
end


function GetPerClassAccuracyRNN(predictions, labels, rho)


  for k = 1,rho do
    labels[k] = torch.squeeze(labels[k]:clone())
    predictions[k] = predictions[k]:clone()    
  end

  local is_all_stop = torch.Tensor(labels[1]:size(1), 1):fill(0)

  for k = 1,rho do
    for i=1,labels[k]:size(1) do
      if labels[k][i] ~= config.nCategories then
        is_all_stop[i] = 1
      end
    end
  end

  local accall = {}
  local correct_sofar = torch.Tensor(labels[1]:size(1), 1):fill(1)
  
  for k = 1,rho do
    
    for i=1,labels[k]:size(1) do
      if labels[k][i] ~= predictions[k][i] then
        correct_sofar[i] = 0;
      end
    end

    local sumnon = 0;
    for i=1,labels[k]:size(1) do
      if is_all_stop[i][1] == 1 then
        sumnon = sumnon + correct_sofar[i][1];
      end
    end

    local tot = 0;
    for i=1,labels[k]:size(1) do
      tot = tot + is_all_stop[i][1];
    end

    table.insert(accall, sumnon / tot)
  end

  return accall
end
    
function GetNMistakeRNN(predictions, labels, rho)

  for k = 1,rho do
    labels[k] = torch.squeeze(labels[k]:clone())
    predictions[k] = predictions[k]:clone()    
  end
  
  local nmist = torch.Tensor(labels[1]:size(1), 1):fill(0)

  
  for i=1,labels[1]:size(1) do
    for k = 1,rho do
      if labels[k][i] ~= predictions[k][i] then
        nmist[i] = nmist[i] + 1
      end
    end
  end

  local accall = {}

  for k = 0,rho do
    local cc = 0
    for i = 1,nmist:size(1) do
      if nmist[i][1] <= k then
        cc = cc + 1
      end
    end
    table.insert(accall, cc / nmist:size(1))
  end

  return accall
end

function ReadListTrainFrames(data_file, input_dir, input_type)
 
  local allfiles = {}  
  
  for line in io.lines(data_file) do 
    local words = {}
    for word in string.gmatch(line,"%w+") do 
      table.insert(words, word) 
    end
    if next(words) ~= nil then
      imid    = words[1]
      objid   = words[2]
      pointid = words[3]
      forceid = words[4]
      magid  = words[5]
    end
    if input_type == "force" then
      for i=1,3 do
        fullfile = imid .. "_" .. objid .. "_" .. pointid .. "_" .. forceid .. "_" ..  tostring(i) .. "_" .. magid .. "x.png";
        table.insert(allfiles, paths.concat(input_dir, fullfile));
      end    
    elseif input_type == "image" then
      fullfile = imid .. ".png"
      table.insert(allfiles, paths.concat(input_dir, fullfile))
    elseif input_type == "mask" then
      --debugger.enter()
      fullfile = imid .. "_" .. objid .. ".png"
      table.insert(allfiles, paths.concat(input_dir, fullfile))
    end

  end

  return allfiles


end


function GetEnableInputTypes(input_config)
  local result = {}
  for input_type, conf in pairs(input_config) do
    if type(conf) == 'table' and conf.enable then
      if config.w_crop and conf.croppable then
        result[ input_type ] = conf.nChannels * 5
      else
        result[ input_type ] = conf.nChannels
      end
    end
  end
  return result
end


function RemoveDups(input)
  local hash = {}
  local res = {}

  for _,v in ipairs(input) do
     if (not hash[v]) then
         res[#res+1] = v -- you could print here instead of saving to result table if you wanted
         hash[v] = true
     end
  end

  return res
end

function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end


function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
    --print(line)
  end
  return lines
end

function split_string(str)
  res = {};
  for i in string.gmatch(str, "%S+") do
    table.insert(res,i);
  end
  return res;
end

