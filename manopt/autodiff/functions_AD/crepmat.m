function Xrepmat = crepmat(X,varargin)
% Replicates an array.
%
% function Xrepmat = crepmat(X,varargin)
%
% This function can be seen as repmat(X,varargin) but is compatible with
% structs with fields real and imag.
%
% See also: manoptAD

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if isstruct(X) && isfield(X,'real')
        Xrepmat.real = repmat(X.real,varargin{:});
        Xrepmat.imag = repmat(X.imag,varargin{:});
        
    elseif isnumeric(X)
        Xrepmat = repmat(X,varargin{:});
        
    else
        ME = MException('crepmat:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
        
    end

end