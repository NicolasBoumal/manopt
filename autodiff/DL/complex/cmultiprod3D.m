function matrix = cmultiprod3D(R1,R2)
    
    if (isnumeric(R1) && isnumeric(R2)) || (isnumeric(R1) && isstruct(R2))
        m = size(R1,1); 
        r = size(R1,3);
        
    elseif isnumeric(R2) && isstruct(R1)
        m = size(R2,1); 
        r = size(R2,3);
        
    elseif isstruct(R1) && isstruct(R2)
        m = size(R1.real,1); 
        r = size(R1.real,3);
    
    else
        ME = MException('cmultiprod3D:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
    
    end
    
    pagetimes(R1,R2);

end