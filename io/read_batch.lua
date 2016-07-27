
function GetAnImageBatch(input_cfg, dataset)
	
	local batchSize   = config.batchSize;
	local target_dim  = config.nPredDim;

  	local target;
  	local images;

	local dataset_size = #dataset;

	local all_input_types = GetEnableInputTypes(config.input_data);
	local nChannels       = GetValuesSum(all_input_types);

    target = torch.CudaTensor(batchSize, target_dim);
    images = torch.CudaTensor(batchSize, nChannels, config.imH, config.imW);
    force_feats = torch.CudaTensor(batchSize, config.force_size)        
    
    
	local cnt = 1
	while cnt <= batchSize do
		local line    = split_string(dataset[config.cur_pointer]);

		local imid    = line[1]
		local objid   = line[2]
		local pointid = line[3]
		local forceid = line[4]
		local magid   = line[5]
		
		local j = 1
		table.sort(config.input_data)
	    for input_type, conf in pairs(config.input_data) do
	      if type(conf) == 'table' and conf.enable then
	      	if input_type == 'image' then
	      		local impath     = paths.concat(conf.dir, imid..'.png');
	      		local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. imid .. '.t7');
				local im = ReadIndividualImage(impath, conf.mean, conf.std, cache_path);				
				images[{{cnt}, {j, j + conf.nChannels-1}, {}, {}}] = im;
				j = j + conf.nChannels

	      	elseif input_type == 'force' then                	      		
	      		local matpath  = paths.concat(conf.dir, imid .. '_' .. objid .. '_' .. pointid .. '_' .. forceid .. '_' .. magid .. 'x.mat');
	      		local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. imid .. '_' .. objid .. '_' .. pointid .. '_' .. forceid .. '_' .. magid .. 'x.t7');
				local force = ReadIndividualForce(matpath, cache_path);										      		
	      		force_feats[{{cnt},{}}] = force.x

	      	elseif input_type == 'mask' then
      			local impath  = paths.concat(conf.dir, imid .. '_' .. objid .. '.png');
      			local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. imid .. '_' .. objid .. '.t7');
				local im = ReadIndividualImage(impath, conf.mean, conf.std, cache_path);
				images[{{cnt}, {j, j + conf.nChannels -1}, {}, {}}] = im;
				j = j + conf.nChannels;

			elseif input_type == 'depth' then
	      		local impath     = paths.concat(conf.dir, imid..'.png');
	      		local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. imid .. '.t7');
				local im = ReadIndividualImage(impath, conf.mean, conf.std, cache_path);				
				images[{{cnt}, {j, j + conf.nChannels-1}, {}, {}}] = im;
				j = j + conf.nChannels
	
	      	end

	      	
	      end
	    end

	   	for i = 18,18+config.rho-1 do  
			target[{{cnt}, {i-17}}] = tonumber(line[i]);
		end    

		cnt = cnt + 1
	    if config.cur_pointer == dataset_size then
	    	config.cur_pointer = 1;
			shuffleList(dataset, 0); 
	    else
	    	config.cur_pointer = config.cur_pointer + 1;
	    end
	end

	local tmp_target = {}
	for i=1,config.rho do
		table.insert(tmp_target, target[{{},{i}}])
	end
	return {images, force_feats}, tmp_target
  
end

function ReadIndividualImage(im_path, mean, std, cache_path)
  	local im;
  	
  	if paths.filep(cache_path) then
  		im = torch.load(cache_path);
  	else
  		im = loadImage(im_path, config.imH, config.imW);			
  		local caffe_mean = mattorch.load(config.caffeFilePath.mean);
  		caffe_mean = caffe_mean.mean_data:transpose(2,3):transpose(1,2)
		
		caffe_mean = caffe_mean[{{},{15,241},{15,241}}]  		
		im = im - caffe_mean

  		torch.save(cache_path, im)
  	end
	
  	return im

end

function ReadIndividualForce(mat_path, cache_path)
  	local force;
  	
  	if paths.filep(cache_path) then
  		force = torch.load(cache_path);
  	else
  		force = mattorch.load(mat_path);
  	
  		torch.save(cache_path, force)
  	end
	
  	return force
end

function GetAUniformImageBatch(input_cfg, dataset)
	
	local batchSize   = config.batchSize;
	local target_dim  = config.nPredDim;

  	local target;
  	local images;


	local dataset_size = #dataset;

	local all_input_types = GetEnableInputTypes(config.input_data);
	local nChannels       = GetValuesSum(all_input_types);

	target = torch.CudaTensor(batchSize, target_dim);
	images = torch.CudaTensor(batchSize, nChannels, config.imH, config.imW);
	force_feats = torch.CudaTensor(batchSize, config.force_size)        

	local cnt = 1
	while cnt <= batchSize do		
		local idx = math.random(dataset_size)

		local line    = split_string(dataset[idx]);

		local imid    = line[1]
		local objid   = line[2]
		local pointid = line[3]
		local forceid = line[4]
		local magid   = line[5]
		
		local j = 1
		table.sort(config.input_data)
	    for input_type, conf in pairs(config.input_data) do
	      if type(conf) == 'table' and conf.enable then
	      	if input_type == 'image' then
	      		local impath     = paths.concat(conf.dir, imid..'.png');
	      		local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. imid .. '.t7');
				local im = ReadIndividualImage(impath, conf.mean, conf.std, cache_path);				
				images[{{cnt}, {j, j + conf.nChannels-1}, {}, {}}] = im;
				j = j + conf.nChannels

	      	elseif input_type == 'force' then
      			local matpath  = paths.concat(conf.dir, imid .. '_' .. objid .. '_' .. pointid .. '_' .. forceid  .. '_' .. magid .. 'x.mat');
      			local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. imid .. '_' .. objid .. '_' .. pointid .. '_' .. forceid .. '_' .. magid .. 'x.t7');
				local force = ReadIndividualForce(matpath, cache_path);										
	      		force_feats[{{cnt},{}}] = force.x;

	      	elseif input_type == 'mask' then
      			local impath  = paths.concat(conf.dir, imid .. '_' .. objid .. '.png');
      			local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. imid .. '_' .. objid .. '.t7');
				local im = ReadIndividualImage(impath, conf.mean, conf.std, cache_path);
				images[{{cnt}, {j, j + conf.nChannels -1}, {}, {}}] = im;
				j = j + conf.nChannels;

			elseif input_type == 'depth' then
	      		local impath     = paths.concat(conf.dir, imid..'.png');
	      		local cache_path = paths.concat(input_cfg.save_dir, input_type .. '_' .. imid .. '.t7');
				local im = ReadIndividualImage(impath, conf.mean, conf.std, cache_path);				
				images[{{cnt}, {j, j + conf.nChannels-1}, {}, {}}] = im;
				j = j + conf.nChannels
	
	      	end

	      	
	      end
	    end

	   	for i = 18,18+config.rho-1 do 
			target[{{cnt}, {i-17}}] = tonumber(line[i]);
		end    

	    cnt = cnt + 1
	end

	local tmp_target = {}
	for i=1,config.rho do
		table.insert(tmp_target, target[{{},{i}}])
	end
	return {images, force_feats}, tmp_target

  
end
