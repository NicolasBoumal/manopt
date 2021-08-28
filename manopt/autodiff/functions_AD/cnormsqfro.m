function Xnormfro = cnormsqfro(X)

    if isstruct(X) && isfield(X,'real')
        Xnormfro = cinnerprodgeneral(X,X);
    
    elseif isnumeric(X)
        if isreal(X) 
            Xnormfro = X(:)'*X(:);
        else
            Xnormfro = sum(real(conj(X(:)).*X(:)));
        end
    
    else
        ME = MException('cnormsqfro:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
        
    end


end