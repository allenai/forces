function compute_mean_std()
  --------------------  COMPUTE MEAN AND STD OF REAL VIDEOS -------------------
  for input_type, input_table in pairs(config.input_data) do
    if type(input_table) == 'table' and input_table.enable then
    
      local meanstdFile = config.input_data.annotation.dir .. '/.meanstd_real_' .. input_type .. '.t7';
      if paths.filep(meanstdFile) then
        local meanstd = torch.load(meanstdFile)
        input_table.mean     = meanstd.mean;
        input_table.std      = meanstd.std;

      else
        local trainDir = input_table.dir;        

        allfiles = ReadListTrainFrames(config.train.datafile, input_table.dir, input_type);
        allfiles = RemoveDups(allfiles)
--        debugger.enter()
        input_table.mean, input_table.std = ComputeMeanStd(1000, allfiles, config.imH, config.imW);        

        local cache = {};
        cache.mean  = input_table.mean;
        cache.std   = input_table.std;
        torch.save(meanstdFile,cache);
      end
    end
  end


end

function LoadCaffeMeanStd(meanFilePath)
  local meanFile = mattorch.load(meanFilePath)
  for input_type, input_table in pairs(config.input_data) do
    if type(input_table) == 'table' and input_table.enable then
      for i=1,3 do
        input_table.mean[i] = meanFile.mean_data:select(3,i):mean() / 255
        input_table.std[i]  = 1/255
      end
    end
  end
end
