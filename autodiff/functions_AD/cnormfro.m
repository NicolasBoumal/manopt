function Xnormfro = cnormfro(X)

    if isstruct(X) && isfield(X,'real')
        Xnormfro = sqrt(cinnerprodgeneral(X,X));
    
    elseif isnumeric(X)
        Xnormfro = sqrt(X(:)'*X(:));
    
    else
        ME = MException('cnormfro:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
        
    end


end