function Xreshape = creshape(X, varargin)
% Reshapes X.
%
% function Xreshape = creshape(X,varargin)
%
% This function can be seen as reshape(X,varargin) but is compatible with
% structs with fields real and imag.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:

    if iscstruct(X)
        Xreshape.real = reshape(X.real, varargin{:});
        Xreshape.imag = reshape(X.imag, varargin{:});
        
    elseif isnumeric(X)
        Xreshape = reshape(X, varargin{:});
        
    else
        ME = MException('creshape:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
        
    end
end
