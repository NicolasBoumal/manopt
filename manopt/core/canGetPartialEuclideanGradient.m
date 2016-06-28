function candoit = canGetPartialEuclideanGradient(problem)
% Checks whether the partial Euclidean gradient can be computed for a problem.
% 
% function candoit = canGetPartialEuclideanGradient(problem)
%
% Returns true if the partial Euclidean gradient of the cost function can
% be computed given the problem description, false otherwise.
%
% See also: getPartialEuclideanGradient canGetPartialGradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 28, 2016.
% Contributors: 
% Change log: 

    candoit = (isfield(problem, 'partialegrad') && isfield(problem, 'ncostterms'));
    
end
