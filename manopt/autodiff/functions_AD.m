function functions_AD()
% This file summarizes the limitations on defining the cost function
% when the user wants to use automatic differentiation to compute the  
% Euclidean gradient and the Euclidean hessian, and provides a list of 
% backup functions to address such issues.
%
% The current implementation of AD is based on Matlab's deep learning
% tool box and dlarray is the key data object which enables functions to  
% compute derivatives via automatic differentiation. However, there is a
% limited number of functions which support dlarray operations, See the
% following website: https://ch.mathworks.com/help/deeplearning/ug/list-
% of-functions-with-dlarray-support.html. Also, operations involving
% complex numbers between dlarrays are not supported yet. So AD fails when 
% the user wants to optimize over complex manifolds. To overcome these
% problems, manopt provides a list of backup functions which are 
% implemented using the functions with AD support so that they have the  
% same functionality but support automatic differentiation,
%
% For the real case, here is a list of functions which are commonly used
% for defing the cost function but do not support dlarrays. They should be 
% replaced by the backup functions which are indicated after the -> symbol.
% trace->ctrace     dot->cdotprod   triu->ctriu     
% norm(...,'fro') ->cnormfro   




% try using preliminar functions in the folder /functions_AD when 
% customizing your cost function. An alternative way is to define one's 
% own preliminary functions which should support both numerical arrays and 
% structures with fields real and imag.





end