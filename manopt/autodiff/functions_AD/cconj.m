function xconj = cconj(x)
% Computes the complex conjugate of x
%
% function xconj = cconj(x)
%
% Returns the complex conjugate of x. The input can be either a numeric 
% array or a struct with fields real and imag.
%
% See also: functions_AD

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if isstruct(x) && isfield(x,'real')
        xconj.real = x.real;
        xconj.imag = -x.imag;
    elseif isnumeric(x)
        xconj = conj(x);
    else
        ME = MException('cconj:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
    end

end