function autogradfunc = autograd(problem, fixedrankflag)
% Apply automatic differentiation to computing the Euclidean gradient
%
% function autogradfunc = autograd(problem)
% function autogradfunc = autograd(problem, fixedrankflag)
%
% Returns an AcceleratedFunction or a function handle which can be used to 
% compute Euclidean gradients. See https://ch.mathworks.com/help/
% deeplearning/ref/deep.acceleratedfunction.html for more descriptions 
% about AcceleratedFunction.
%
% Note: to evaluate the Euclidean gradient of a certain point x(x should be
% of type dlarray), call dfeval(autogradfunc,x) instead of autogradfunc(x).
%
% See also: manoptAD, egradcompute, costgradcompute

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 
%
% To do: Add AD to fixedTTrankfactory, fixedranktensorembeddedfactory
% and the product manifold which contains fixedrankembeddedfactory
% or anchoredrotationsfactory
    
    % Check availability 
    assert(isfield(problem,'M') && isfield(problem,'cost'),...
    'problem structure must contain the fields M and cost.');
    assert(exist('dlarray', 'file') == 2, ['Deep learning tool box is '... 
    'needed for automatic differentiation'])
    
    % Set fixedrankflag to false if the manifold struct is not 
    % fixed(multilinear)-rank matrices or tensors with an embedded geometry
    % or tensors of fixed Tensor Train (TT) rank
    if ~exist('fixedrankflag', 'var')|| isempty(fixedrankflag)
        fixedrankflag = false;
    end

    % Obtain the euclidean gradient function via AD
    costfunction = problem.cost;
    % Set fixedrankflag to true if the manifold is fixed-rank matrices with
    % an embedded geometry. The other two cases are not implemented yet.
    if fixedrankflag
        % AcceleratedFunction can lead to a slow down in this case
        autogradfunc = @(x,A,B) autogradfuncinternelfixedrankembedded(x,A,B);
    else
        func = @autogradfuncinternal;
        % accelerate 
        try
            autogradfunc = dlaccelerate(func); % Introduced in Matlab 2021a
            clearCache(autogradfunc);
        catch
            warning('manopt:dlaccelerate', ...
                    ['Function dlaccelerate is not available:\nPlease ' ...
                     'upgrade to Matlab 2021a or later and the latest deep\nlearning ' ...
                     'toolbox version if possible.\nMeanwhile, auto-diff ' ...
                     'may be somewhat slower.\n The hessian is not available as well.\n' ...
                     'To disable this warning: warning(''off'', ''manopt:dlaccelerate'')']);
            autogradfunc = func;
        end
    end
    
    % define Euclidean gradient function
    function [y, egrad] = autogradfuncinternal(x)
            
        y = costfunction(x);
        % In case that the user forgot to take the real part of the cost
        % when dealing with complex problems with Matlab R2021a or earlier, 
        % take the real part for AD
        if iscstruct(y)
            y = creal(y);
        end
        
        % Call dlgradient to compute the Euclidean gradient. by default, 
        % 'RetainData' and 'EnableHigherDerivatives' are set to false
        egrad = dlgradient(y, x);
        
        % in case that the user is optimizing over anchoredrotationsfactory
        % egrad of anchors with indices in A should be zero
        problem_name = problem.M.name();
        if (contains(problem_name,'Product rotations manifold') &&..., 
            contains(problem_name,'anchors') &&...,
            ~startsWith(problem_name,'Product manifold'))
            A = findA_anchors(problem);
            egrad(:, :, A) = 0;
        end
    end
    
    % fixedrankembeddedfactory part
    % obtain the product of egrad and V and the product of egrad
    % transpose and U by differentiating g1 and g2 w.r.t A and B
    function [g1, egrad] = autogradfuncinternelfixedrankembedded(x, A, B)
        X1.U = A; X1.S = eye(size(x.S,1)); X1.V = x.V;
        X2.U = x.U; X2.S = eye(size(x.S,1)); X2.V = B;
        g1 = costfunction(X1); g2 = costfunction(X2);
        egrad.A = dlgradient(g1,A);  egrad.B = dlgradient(g2,B);
    end
    
end
