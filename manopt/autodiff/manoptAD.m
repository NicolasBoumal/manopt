function manoptAD()
% This file provides certain comments regarding the use of automatic
% differentiation in manopt. It summarizes the limitations to define the  
% cost function when the user wants to use automatic differentiation to 
% provide the information of the gradient and the hessian. This file also 
% provides a list of backup functions to address such issues.
%
% The current implementation of AD is based on Matlab's deep learning
% tool box in which dlarray is the key data object which enables functions 
% to compute derivatives via automatic differentiation. However, there is a
% couple of functions which do not support dlarray operations. See the
% following official website: 
% https://ch.mathworks.com/help/deeplearning/ug/list-of-functions-with-dlarray-support.html. 
% Also, operations containing complex numbers between dlarrays are only 
% introduced in Matlab R2021b or later. So AD fails when the user wants to 
% optimize over complex manifolds when the Matlab installed is R2021a or 
% earlier. In these cases, manopt provides a list of backup functions which 
% are implemented using the functions with AD support so that they have the 
% same functionality but support automatic differentiation.
% 
% For Matlab R2021b or later: Complex numbers are supported. Users only
% need to take care of those functions which do not support dlarray. Here 
% is the list of common functions without AD support and they should be  
% replaced by the corresponding backup functions when specifying the cost.
% trace->ctrace
% triu->ctriu
% tril->ctril     
% norm(...,'fro') ->cnormfro
% norm(...,'fro')^2 ->cnormsqfro 
% diag->cdiag
% multiscale->cmultiscale
% All the other multi*** functions in manopt are supported.
% Moreover, bsxfun is not supported. The user may have to translate it
% into repmat and point-wise expressions. Concatenating arrays along the 
% third or higher dimension cat(A,B,3+) is not supported for AD.
% Besides, the matrix functions sqrtm, logm, expm, eig, svd, det, cumsum, 
% movsum, cumprod, \, inv, mod, rem, vecnorm, bandwidth are not supported. 
%
% For Matlab R2021a or earlier, in addition to the functions mentioned  
% above, dot is not supported. The element-wise multiplication can be 
% replaced by cdotprod. Moreover, to deal with complex problems, manopt 
% converts complex numbers into a structure with fields real and imag.
% See mat2dl_complex and dl2mat_complex for more details. In this case,
% the user should try using the basic functions in the folder 
% /functions_AD when defining the cost function. An alternative way is 
% to define one's own basic functions. These functions should 
% accept both numerical arrays and structures with fields real and imag. 
% See cprod and complex_example_AD as examples.
%
% See also: preprocessAD, complex_example_AD

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:

    fprintf('This file is just a documentation.');

end