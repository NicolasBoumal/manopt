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
    
    A = creshape(R1,m,m,1,r);
    B = creshape(R2,1,m,m,r);
    A = crepmat(A,1,1,m,1);
    B = crepmat(B,m,1,1,1);
    C = csum(cdotprod(A,B),2);
    matrix = creshape(C,m,m,r,1);

end