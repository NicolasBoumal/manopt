function xsum = csum(x,n)
% Sum of elements of x.
%
% function xsum = csum(x,n)
%
% This function can be seen as sum(x,n) but is compatible with
% structs with fields real and imag.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:    

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
        
end