function manoptADhelp()
% Automatic differentation (AD) in Manopt requires the following:
%   Matlab version R2021a or later.
%   Deep Learning Toolbox version 14.2 or later.
%
% First, read the documentation of manoptAD by typing:
%
%   help manoptAD
%
% The comments here provide further information about how to use AD. These
% comments are necessary because of certain limitations both of the AD
% capabilities of the Deep Learning Toolbox and of Manopt's ability to
% handle certain delicate geometries, e.g., fixed-rank matrices, with AD.
%
% The basic usage of AD in Manopt goes as follows:
%
%   problem.M = ...; %call a factory to get a manifold structure
%   problem.cost = @(x) ...; % implement your cost function
%   problem = manoptAD(problem); % ask AD to figure out gradient + Hessian
%   ...; % call a solver on problem, e.g., x = trustregions(problem);
%
% The current implementation of AD is based on Matlab's Deep Learning
% Toolbox. The latter relies on dlarray to represent data (arrays).
% While this works very well, unfortunately, there are a few limitations.
% For example, certain functions do not support dlarray.
% See the following official website for a list of compatbile functions: 
%
%   https://ch.mathworks.com/help/deeplearning/ug/list-of-functions-with-dlarray-support.html.
%
% Moreover, dlarray only supports complex variables starting with
% Matlab R2021b.
%
% To handle complex numbers in R2021a and earlier, and also to handle
% functions that are not supported by dlarray at this moment, Manopt
% provides a limited collection of backup functions which are compatible
% with AD. Explicitly, the left column below lists commonly used functions
% which are not supported by dlarray at this time, and the right column
% lists corresponding replacement functions that can be used:
%
%     trace                 ctrace
%     diag                  cdiag
%     triu                  ctriu
%     tril                  ctril
%     norm(..., 'fro')      cnormfro
%     norm(..., 'fro')^2    cnormsqfro
%     multiscale            cmultiscale
%
% All the other multi*** functions in Manopt support AD.
%
% Moreover, bsxfun is not supported. The user may have to translate it
% into repmat and point-wise expressions.
% Concatenating arrays along the third or higher dimension such as in
% cat(A, B, 3+) is not supported for AD.
% The matrix functions sqrtm, logm, expm, eig, svd, det, cumsum,  movsum,
% cumprod, \, inv, mod, rem, vecnorm, bandwidth are not supported.
%
% For Matlab R2021a or earlier, in addition to the functions mentioned  
% above, dot is not supported.
% Element-wise multiplication can be replaced by cdottimes.
%
% To deal with complex variables in R2021a and earlier, Manopt converts
% complex arrays into a structure with fields real and imag.
% See mat2dl_complex and dl2mat_complex for more details. In this case,
% the user should try using the basic functions in the folder 
%
%   /manopt/autodiff/functions_AD
%
% when defining the cost function. An alternative way is to define one's
% own basic functions. These functions should accept both numeric arrays
% and structures with fields real and imag.
% See cprod and complex_example_AD as examples.
%
% See also: manoptAD complex_example_AD

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:

    fprintf('The file manoptADhelp is just for documentation: run ''help manoptADhelp''.\n');

end
