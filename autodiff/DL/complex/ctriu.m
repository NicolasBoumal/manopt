function Xtriu = ctriu(X,k)

    switch nargin
        case 1
            if isstruct(X) && isfield(X,'real')
                index0 = find(triu(ones(size(X.real)))==0);
                Xtriu = X;
                Xtriu.real(index0) = 0;
                Xtriu.imag(index0) = 0;
        
            elseif isnumeric(X) && ~isdlarray(X)
                Xtriu = triu(X);
            
            elseif isdlarray(X)
                Xtriu = dlarray(zeros(size(X)));
                index1 = find(triu(ones(size(X)))==1);
                Xtriu(index1) = X(index1);
                
            else
                ME = MException('ctriu:inputError', ...
                'Input does not have the expected format.');
                throw(ME);
            end
        case 2
            if isstruct(X) && isfield(X,'real')
                index0 = find(triu(ones(size(X.real)),k)==0);
                Xtriu = X;
                Xtriu.real(index0) = 0;
                Xtriu.imag(index0) = 0;
        
            elseif isnumeric(X) && ~isdlarray(X)
                Xtriu = triu(X,k);
            
            elseif isdlarray(X)
                Xtriu = dlarray(zeros(size(X)));
                index1 = find(triu(ones(size(X)),k)==1);
                Xtriu(index1) = X(index1);
                
            else
                ME = MException('ctriu:inputError', ...
                'Input does not have the expected format.');
                throw(ME);
            end
    otherwise
        error('Too many input arguments.');
    end
end

