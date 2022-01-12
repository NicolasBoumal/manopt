function y = sinxoverx(x)
% Computes sin(x) ./ x entrywise, such that sin(0) / 0 yields 1.
%
% function y = sinxoverx(x)
%
% The function sinc from the Signal Processing Toolbox is related to this
% function via sinc(x) = sinxoverx(pi*x).
% 
% See also: sinc

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Jan. 12, 2022.
% Contributors:
% Change log:

    y = sin(x) ./ x ;  % By default, divisions by zero trigger no warnings.
    y(x == 0) = 1;

end
