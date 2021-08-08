function autogradfunc = autograd(problem)
% Apply automatic differentiation to computing Euclidean gradient
%
% function autogradfunc = autograd(problem)
%
% Returns an AcceleratedFunction which is used to compute Euclidean 
% gradients. See https://ch.mathworks.com/help/deeplearning/ref/deep.
% acceleratedfunction.html for more descriptions about AcceleratedFunction.

% Note: to evaluate the Euclidean gradient of a certain point x(x should be
% of type dlarray), call dfeval(autogradfunc,x) instead of autogradfunc(x).

% See also: egradcompute
    
    % check availability 
    assert(isfield(problem,'M') && isfield(problem,'cost'),...,
    'problem structure must contain fields M and cost.');
    assert(exist('dlarray', 'file') == 2, ['Deep learning tool box is '... 
    'needed for automatic differentiation'])
    
    % obtain the Euclidean gradient function using AD
    costfunction = problem.cost;
    func = @(x) autogradfuncinternel(costfunction,x);
    % accelerate 
    autogradfunc = dlaccelerate(func);
    
    % define Euclidean gradient function
    function [y egrad] = autogradfuncinternel(costfunction,x)
       
        y = costfunction(x);
        
        % in case that the user forgot to take the real part of the cost
        % when dealing with complex problems, take the real part for AD
        if isstruct(y) && isfield(y,'real')
            y = creal(y);
        end
        
        % call dlgradient to compute the Euclidean gradient. by default, 
        % 'RetainData' and 'EnableHigherDerivatives' are set to false
        egrad = dlgradient(y,x);
        
        % in case that the user is optimizing over anchoredrotationsfactory
        % egrad of anchors with indices in A should be zero
        if (contains(problem.M.name(),'Product rotations manifold') &&..., 
            contains(problem.M.name(),'anchors'))
        A = problem.M.A;
        egrad(:, :, A) = 0;
        end
        
    end
end