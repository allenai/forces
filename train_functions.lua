log('Loading Train Functions ... ')



function train()
  input_cfg = config.train;
  input_cfg_test = config.test;
  ----------------------------
  dataset = lines_from(input_cfg.datafile);

  dataset_test = lines_from(input_cfg_test.datafile);

  local batchSize = config.batchSize;
  
  
  for iter=1,config.nIter do
    ---- load one batch
    local tic= os.clock()
    local TrInput, TrTarget = GetAnImageBatch(input_cfg, dataset)    
    local toc = os.clock() - tic;
    log('loading time :' .. tostring(toc))

    -------- train the network--------------
    model.learningRate = model:LearningRateComp(iter);

    local acc, per_class, loss = model:TrainOneBatch(TrInput,TrTarget);

    if (iter % 10) == 0 then
      local  tic = os.clock()
      collectgarbage();
      local toc = os.clock() - tic;
      print("garbage collection :", toc)
    end


    if (iter % config.nDisplay) == 0 then
      log(('Iter = %d | Train Loss = %f\n'):format(iter,loss));

      for i = 1,config.rho do
        if i == config.rho then
          log(('Train Accuracy -- Global [%d] = %f \n'):format(i, acc[i]));
        else      
          log(('Train Accuracy -- Global [%d] = %f '):format(i, acc[i]));
        end
      end

      for i = 1,config.rho do
        if i == config.rho then
          log(('Train Accuracy -- Per_Class [%d] = %f \n'):format(i, per_class[i]));
        else
          log(('Train Accuracy -- Per_Class [%d] = %f '):format(i, per_class[i]));
        end
      end

    end

    if (iter % config.nEval) == 0 then
      local TeInput, TeTarget = GetAUniformImageBatch(input_cfg_test, dataset_test);      
      local acc, per_class, loss = model:EvaluateOneBatch(TeInput,TeTarget);
      log(('Testing ---------> Iter = %d | Test Loss = %f\n'):format(iter,loss));

      for i = 1,config.rho do
        if i == config.rho then
          log(('Test Accuracy -- Global [%d] = %f \n'):format(i, acc[i]));
        else
          log(('Test Accuracy -- Global [%d] = %f '):format(i, acc[i]));
        end
      end

      for i = 1,config.rho do
        if i == config.rho then
          log(('Test Accuracy -- Per_Class [%d] = %f \n'):format(i, per_class[i]));
        else
          log(('Test Accuracy -- Per_Class [%d] = %f '):format(i, per_class[i]));
        end
      end

    end
    
    if (iter % config.saveModelIter) == 0 then
      local fileName = 'Model_iter_' .. iter .. '.t7';
      log('Saving NN model in ----> ' .. paths.concat(config.logDirectory, fileName) .. '\n');
      model:SaveModel(paths.concat(config.logDirectory, fileName));
      config.imgFilenamesLog:flush()
    end

  end
end


---------------------------------------------------------
function test()
  input_cfg = config.test;
  ----------------------------
  dataset = lines_from(input_cfg.datafile);

  local batchSize = config.batchSize;

  local all_predictions
  local all_targets

  for iter=1,config.nIter do
    ---- load one batch
    local tic= os.clock()
    local TeInput, TeTarget = GetAnImageBatch(input_cfg, dataset);
    
    local toc = os.clock() - tic;
    log('loading time :' .. tostring(toc))

    if (iter % 10) == 0 then
      local  tic = os.clock()
        collectgarbage();
      local toc = os.clock() - tic;
      print("garbage collection :", toc)
    end
    local acc, per_class, loss, acc_all, predicts = model:EvaluateOneBatch(TeInput,TeTarget);

    if not all_predictions then
      all_predictions = predicts
    else
      for i = 1, config.rho do
        all_predictions[i] = torch.cat(all_predictions[i], predicts[i], 1)
      end
    end

    if not all_targets then
      all_targets = TeTarget
    else
      for i = 1, config.rho do
        all_targets[i] = torch.cat(all_targets[i], TeTarget[i], 1)
      end
    end

  end
  local fname = input_cfg.outfile

  results_pred = torch.Tensor(all_predictions[1]:size(1),config.rho)
  results_gt = torch.Tensor(all_targets[1]:size(1),config.rho)

  log("Saving all predictions at " .. fname)
  for i = 1,config.rho do
    results_pred[{{},{i}}] = all_predictions[i]:double()
    results_gt[{{},{i}}] = all_targets[i]:double()
  end


  vars = {preds = results_pred, targets = results_gt}
  log('Saving results in ----> ' .. fname .. '\n');
  mattorch.save(fname, vars)
  
end




