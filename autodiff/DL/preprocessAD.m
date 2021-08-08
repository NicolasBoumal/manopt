function problem = preprocessAD(problem) 
% Preprocess automatic differentiation for the problem structure

% function problem = preprocessAD(problem)

% Check if automatic differentiation provided in the deep learning tool
% box can be applied to computing the euclidean gradient and the euclidean
% hessian given the manifold and cost function described in the problem
% structure (if no egrad and ehess is provided). If AD fails for some
% reasons, the original problem structure is returned and the approx. of
% gradient and hessian will then be used as usual. Otherwise, the problem
% structure with two additional fields: egrad and ehess is returned.

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

% See also: mat2dl_complex, autograd, egradcompute, ehesscompute

% To do: Add AD to fixedrankembeddedfactory, fixedrankMNquotientfactory,
% fixedranktensorembeddedfactory, symfixedrankYYfactory, 
% symfixedrankYYcomplexfactory. Add the gpu feature to AD.

%% Check if AD can be applied to the manifold and cost function
    
    assert(isfield(problem,'M') && isfield(problem,'cost'),...,
    'the problem structure must contain fields M and cost.');
    
    % if gradient and hessian information is provided already, just return
    if (isfield(problem,'egrad') && isfield(problem,'ehess'))..., 
            || (isfield(problem,'egrad') && isfield(problem,'hess'))...,
            || (isfield(problem,'grad') && isfield(problem,'ehess'))...,
            || (isfield(problem,'grad') && isfield(problem,'hess'))...,
            || (isfield(problem,'costgrad') && isfield(problem,'ehess'))...,
            || (isfield(problem,'costgrad') && isfield(problem,'hess'))
        return 
    % AD does not support euclideansparsefactory so far.
    elseif contains(problem.M.name(),'sparsity')
         warning(['Auto differentiation currently does not support '...
                    'sparse matrices']);
        return
    % check availability.
    elseif ~(exist('dlarray', 'file') == 2)
        sprintf(['It seems the Deep learning tool box is not installed.', ...
         '\nIt is needed for automatic differentiation \n']);
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
                    warning(['Auto differentiation failed. '...
                    'Cost function contains complex numbers. Check if '...
                    'it works for both numerical arrays and structures with'...
                    ' fields real and imag']);
                    return
                end
                % if no error appears, set complexflag to true
                complexflag = true;
            else
                % if the error is not related to complex number, then it
                % must be the problem of defining cost function
                 warning(['Auto differentiation failed. '...
                    'Problem defining cost function. Check the list of '...
                    'functions with AD support on the following website.'...
                    'https://ww2.mathworks.cn/help/deeplearning/ug/'...
                    'list-of-functions-with-dlarray-support.html']);
                return   
            end
        end                   
    end
%% Use AD to compute euclidean gradient and euclidean hessian

    problem.autogradfunc = autograd(problem);
    problem.egrad = @(x) egradcompute(problem,x,complexflag);
    problem.costgrad = @(x) costgradcompute(problem,x,complexflag);
    problem.ehess = @(x,xdot,store) ehesscompute(problem,x,xdot,store,complexflag);
    
    % some functions are not supported to differentiate with AD.
    % e.g.cat(3,A,B). Check availablility of egrad, if not, remove fields
    % egrad and ehess.
    try 
        egrad = problem.egrad(x);
    catch
        warning(['Auto differentiation failed. '...
                    'Problem defining cost function. Check the list of '...
                    'functions with AD support on the following website.'...
                    'https://ww2.mathworks.cn/help/deeplearning/ug/'...
                    'list-of-functions-with-dlarray-support.html']);
        problem = rmfield(problem,'egrad');
        problem = rmfield(problem,'ehess');
        return
    end
    
    warning(['It seems no gradient was provided. '...
                    'Automatic differentiation is used to compute egrad and ehess']);
    
end