function candoit = canGetHessian(problem)
% Checks whether the Hessian can be computed for a problem structure.
%
% function candoit = canGetHessian(problem)
%
% Returns true if the Hessian of the cost function can be computed given
% the problem description, false otherwise.
%
% See also: canGetCost canGetDirectionalDerivative canGetGradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    candoit = isfield(problem, 'hess');
    
end
