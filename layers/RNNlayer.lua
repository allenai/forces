function rnn(inDim, hidDim, outDim , rho, shared)

     local function SelectRange(m,n)
     	 local C = nn.ConcatTable()
     	 for i= m,n do
     	    C:add(nn.SelectTable(i))
     	 end
     	 return C
     end

     local function JoinTable_firstTwo(dim)
     	 local C = SelectRange(1,2)
     	 return nn.Sequential():add(C):add(nn.JoinTable(dim))
     end

     local function joinUnit()  
     	return nn.Sequential():add(JoinTable_firstTwo(2)):add(nn.Linear(inDim+hidDim,hidDim)):add(nn.ReLU(true))
     end
     local function maxUnit() 
       return nn.Sequential():add(nn.SelectTable(2)):add(nn.Linear(hidDim,outDim))
     end

     local rnnUnit = nn.Sequential()
     for i=1,rho do
         hidUnit = nn.ConcatTable()
                      :add(nn.SelectTable(1))
                      :add(joinUnit())
                      :add(maxUnit())
        if i>1 then
        	for j=1,i-1 do
        		hidUnit:add(nn.SelectTable(2+j))
        	end     
        end 
        rnnUnit:add(hidUnit)
     end
	 rnnUnit:add(SelectRange(3,2+rho))
	 
	 if shared == true then	 	
	 	local linUnits = rnnUnit:findModules('nn.Linear')
	 	for i=3,#linUnits,2 do
        	linUnits[i] = linUnits[1];
         	linUnits[i+1] =linUnits[2]    
     	end
     end

	 return rnnUnit
end
