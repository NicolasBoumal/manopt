function [cost, grad] = costgradcompute(problem, x, complexflag)
% Computes the cost and the gradient at x via AD in one call 
%
% function [cost, egrad] = costgradcompute(problem, x, complexflag)
%
% Returns the cost and the gradient of the cost function described in 
% the problem structure at the point x.
%
% Note: the problem structure must contain the field autogradfunc.
% autogradfunc should be either an AcceleratedFunction or a function handle  
% which contains dlgradient. x is a point on the target manifold. 
% complexflag is bool variable which indicates whether or not the problem 
% described in autogradfunc involves complex numbers and meanwhile the
% Matlab version is R2021a or earlier.
%
% See also: manoptAD autograd mat2dl mat2dl_complex dl2mat dl2mat_complex
    
% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 
%
% To do: Add AD to fixedTTrankfactory, fixedranktensorembeddedfactory
% and the product manifold which contains fixedrankembeddedfactory
% or anchoredrotationsfactory

    assert(isfield(problem,'autogradfunc'),['the problem structure must'...,
        ' contain the field autogradfunc, see autograd.'])
    % convert x into dlarrays to prepare for AD
    if complexflag == true
        dlx = mat2dl_complex(x);
    else
        dlx = mat2dl(x);
    end

    % In Matlab R2021b Prerelease, AcceleratedFunction can only accept
    % the input with a fixed data structure. If the representation of 
    % a point on the manifold varies when running a certain algorithm, 
    % the AcceleratedFunction then fails to work properly. A special case   
    % is that AcceleratedFunction is sensitive to the order in which the 
    % fields of the structure have been defined. If a point on a manifold 
    % is represented as a structure and meanwhile the order of the fields 
    % defined in the retr and the rand functions in a manifold factory are 
    % inconsistent, an error will occur. In this case, the old cache should 
    % be cleared in order to accept the new input.
    if isa(problem.autogradfunc,'deep.AcceleratedFunction')
        try
            % compute egrad according to autogradfunc
            [cost, egrad] = dlfeval(problem.autogradfunc, dlx);
        catch
            % clear the old cache
            clearCache(problem.autogradfunc);
            [cost, egrad] = dlfeval(problem.autogradfunc, dlx);
            warning('manopt:AD:cachedlaccelerate', ...
            ['The representation of points on the manifold is inconsistent.\n'...
            'AcceleratedFunction has to clear its old cache to accept the new '...
            'representation of the input.\nPlease check the consistency when '...
            'writing the manifold factory.\n'...
            'To disable this warning: warning(''off'', ''manopt:AD:cachedlaccelerte'')']);            
        end
     else
        [cost, egrad] = dlfeval(problem.autogradfunc, dlx);
     end

    % convert egrad back to numerical arrays
    if complexflag == true
        egrad = dl2mat_complex(egrad);
    else
        egrad = dl2mat(egrad);
    end
    
    % convert egrad to rgrad
    grad = problem.M.egrad2rgrad(x, egrad);
    % convert cost back to a numeric format
    cost = dl2mat(cost);
    
end
