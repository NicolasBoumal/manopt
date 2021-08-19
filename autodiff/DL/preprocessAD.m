function problem = preprocessAD(problem) 
% Preprocess automatic differentiation for the problem structure

% function problem = preprocessAD(problem)

% Check if automatic differentiation provided in the deep learning tool
% box can be applied to computing the euclidean gradient and the euclidean
% hessian given the manifold and cost function described in the problem
% structure (if no gradient or hessian information is provided). If AD 
% fails for some reasons, the original problem structure is returned and 
% the approx. of gradient and hessian will then be used as usual. 
% Otherwise, the problem structure with additional fields: egrad, costgrad 
% and ehess is returned.

% In the case that the manifold is the set of fixed-rank matrices with 
% an embedded geometry, it is more efficient to compute the Riemannian 
% gradient directly. However, computing the exact Riemannian Hessian by 
% vector product via AD is currently not supported. By calling 
% preprocessAD, the problem struct with additional fields grad and costgrad
% is returned.

% Note: The current functionality of AD relies on Matlab's deep learning
% tool box, which has the inconvenient effect that we cannot control the
% limitations. Firstly, AD does not support sparse matrices so far. Try 
% converting sparse arrays into full arrays in the cost function. Secondly, 
% math operations involving complex numbers are currently not supported for
% dlarray. To deal with complex problems, try using preliminary
% functions in the folder /complexfunctions when customizing your cost
% function. An alternative way is to define one's own preliminary functions
% which should support both numerical arrays and structures with fields
% real and imag. Thirdly, check the list of functions with AD support
% when defining the cost function. See the website: https://ww2.mathworks.
% cn/help/deeplearning/ug/list-of-functions-with-dlarray-support.html
% To run AD on GPU, set gpuflag = true in the problem structure and store 
% related arrays on GPU as usual.

% See also: mat2dl_complex, autograd, egradcompute, ehesscompute
% gradcomputefixedrankembedded, costgradcomputefixedrankembedded

% To do: Add AD to fixedranktensor

%% Check if AD can be applied to the manifold and the cost function
    
    assert(isfield(problem,'M') && isfield(problem,'cost'),...,
    'the problem structure must contain fields M and cost.');
    
    % if the gradient and hessian information is provided already, return
    if  (isfield(problem,'egrad') && isfield(problem,'ehess'))..., 
            || (isfield(problem,'egrad') && isfield(problem,'hess'))...,
            || (isfield(problem,'grad') && isfield(problem,'ehess'))...,
            || (isfield(problem,'grad') && isfield(problem,'hess'))...,
            || (isfield(problem,'costgrad') && isfield(problem,'ehess'))...,
            || (isfield(problem,'costgrad') && isfield(problem,'hess'))
        return 
    % AD does not support euclideansparsefactory so far.
    elseif contains(problem.M.name(),'sparsity')
         warning(['Automatic differentiation currently does not support '...
                    'sparse matrices']);
        return
    % check availability.
    elseif ~(exist('dlarray', 'file') == 2)
        sprintf(['It seems the Deep learning tool box is not installed.', ...
         '\nIt is needed for automatic differentiation. \n']);
        return
    else 
        % complexflag is used to detect if the problem defined contains
        % complex numbers.
        complexflag = false;
        % check if AD can be applied to the cost function by passing a
        % point on the manifold to problem.cost.
        x = problem.M.rand();
        try
            dlx = mat2dl(x);
            costtestdlx = problem.cost(dlx);
        catch ME
            % detect complex number by looking up error message
            if (strcmp(ME.identifier,'deep:dlarray:ComplexNotSupported'))
                try
                    dlx = mat2dl_complex(x);
                    costtestx = problem.cost(x);
                    costtestdlx = problem.cost(dlx);
                catch
                    warning(['Automatic differentiation failed. '...
                    'Cost function contains complex numbers. Check if '...
                    'it works for both numerical arrays and structures with'...
                    ' fields real and imag']);
                    return
                end
                % if no error appears, set complexflag to true
                complexflag = true;
            else
                % if the error is not related to complex number, then it
                % must be the problem of defining the cost function
                warning(['Automatic differentiation failed. '...
                    'Problem defining the cost function. '...
                    '<a href = "https://ww2.mathworks.cn/help/deeplearning'...
                    '/ug/list-of-functions-with-dlarray-support.html">'...
                    'Check the list of functions with AD support.</a>']);
                return   
            end
        end                   
    end
%% compute the euclidean gradient and the euclidean hessian via AD

    % check if the manifold struct is fixed-rank matrices 
    % with an embedded geometry. for fixedrankembedded factory, 
    % only the Riemannian gradient can be computed via AD so far.
    fixedrankflag = 0;
    if (sum(isfield(x,{'U','S','V'}))==3) &&..., 
        (contains(problem.M.name(),'rank'))
        % set the fixedrankflag to 1 to prepare for autgrad
        fixedrankflag = 1;
        % if no gradient information is provided, compute grad using AD
        if ~isfield(problem,'egrad') && ~isfield(problem,'grad')...,
            && ~isfield(problem,'costgrad')
            problem.autogradfunc = autograd(problem,fixedrankflag);
            problem.grad = @(x) gradcomputefixedrankembedded(problem,x);
            problem.costgrad = @(x) costgradcomputefixedrankembedded(problem,x);
        else
        % computing the exact hessian via AD is currently not supported
            return
        end
    end
    
    % for other manifolds, provide egrad and ehess via AD. manopt can 
    % get grad and hess automatically through egrad2rgrad and ehess2rhess
    
    % if only the hessian information is provided, compute egrad 
    % hessianflag indicates whether or not ehess or hess has provided already 
    hessianflag = false;
    if ~isfield(problem,'egrad') && ~isfield(problem,'grad')...,
            && ~isfield(problem,'costgrad') && (isfield(problem,'ehess')...,
            || isfield(problem,'hess'))
        
        problem.autogradfunc = autograd(problem);
        problem.egrad = @(x) egradcompute(problem,x,complexflag);
        problem.costgrad = @(x) costgradcompute(problem,x,complexflag);
        hessianflag = true;
        
    % if only the gradient information is provided, compute ehess     
    elseif ~isfield(problem,'ehess') && ~isfield(problem,'hess')...,
            && (isfield(problem,'costgrad') || isfield(problem,'grad')...,
            || isfield(problem,'egrad')) && (fixedrankflag == 0)
    
        problem.ehess = @(x,xdot,store) ehesscompute(problem,x,xdot,store,complexflag);
        
    % otherwise compute both egrad and ehess via automatic differentiation      
    elseif fixedrankflag == 0
        problem.autogradfunc = autograd(problem);
        problem.egrad = @(x) egradcompute(problem,x,complexflag);
        problem.costgrad = @(x) costgradcompute(problem,x,complexflag);
        problem.ehess = @(x,xdot,store) ehesscompute(problem,x,xdot,store,complexflag);
    end
    
%% check whether the cost function can be differentiated or not

    % some functions are not supported to be differentiated with AD in the
    % deep learning tool box. e.g.cat(3,A,B). Check availablility of egrad,
    % if not, remove relevant fields such as egrad and ehess.
    
    if isfield(problem,'autogradfunc') && (fixedrankflag == 0)
        try 
            egrad = problem.egrad(x);
        catch
            warning(['Automatic differentiation failed. '...
                    'Problem defining the cost function. '...
                    '<a href = "https://ww2.mathworks.cn/help/deeplearning'...
                    '/ug/list-of-functions-with-dlarray-support.html">'...
                    'Check the list of functions with AD support.</a>']);
            problem = rmfield(problem,'egrad');
            if ~hessianflag
                problem = rmfield(problem,'ehess');
            end
            return
        end
    % if only egrad or grad is provided, check ehess
    elseif ~isfield(problem,'autogradfunc') && (fixedrankflag == 0) && ~hessianflag
        % randomly generate a point in the tangent space at x
        xdot = problem.M.randvec(x);
        store = struct();
        try 
            ehess = problem.ehess(x,xdot,store);
        catch
            warning(['Automatic differentiation failed. '...
                    'Problem defining the cost function. '...
                    '<a href = "https://ww2.mathworks.cn/help/deeplearning'...
                    '/ug/list-of-functions-with-dlarray-support.html">'...
                    'Check the list of functions with AD support.</a>']);
            problem = rmfield(problem,'ehess');
            return
        end
    % check the case of fixed rank matrices endowed with an embedded geometry 
    elseif isfield(problem,'autogradfunc') && fixedrankflag == 1
        try 
            grad = problem.grad(x);
        catch
            warning(['Automatic differentiation failed. '...
                    'Problem defining the cost function. '...
                    '<a href = "https://ww2.mathworks.cn/help/deeplearning'...
                    '/ug/list-of-functions-with-dlarray-support.html">'...
                    'Check the list of functions with AD support.</a>']);
            problem = rmfield(problem,'grad');
            return
        end
        
    end
    
    
end