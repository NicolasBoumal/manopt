function xsum = csum(x,n)
    
    if isstruct(x) && isfield(x,'real')
        if nargin==1
            xsum.real = sum(x.real);
            xsum.imag = sum(x.imag);
        elseif nargin==2
            xsum.real = sum(x.real,n);
            xsum.imag = sum(x.imag,n);
        end
    elseif isnumeric(x)
        if nargin==1
            xsum = sum(x);
        elseif nargin==2
            xsum = sum(x,n);
        end
    else
        ME = MException('csum:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
        





end