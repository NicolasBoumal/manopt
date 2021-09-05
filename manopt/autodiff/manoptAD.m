function problem = manoptAD(problem, flag) 
% Preprocess automatic differentiation for a manopt problem structure
%
% function problem = manoptAD(problem)
% function problem = manoptAD(problem, 'nohess')
% function problem = manoptAD(problem, 'hess')
%
% Given a manopt problem structure with problem.cost and problem.M defined,
% this tool adds the following fields to the problem structure:
%   problem.egrad
%   problem.costgrad
%   problem.ehess
%
% A field problem.autogradfunc is also created for internal use.
%
% The fields egrad and ehess correspond to Euclidean gradients and Hessian.
% They are obtained through automatic differentation of the cost function.
% Manopt converts them into Riemannian objects in the usual way via the
% manifold's M.egrad2rgrad and M.ehess2rhess functions, automatically.
%
% As an optional second input, the user may specify the flag string to be:
%   'nohess' -- in which case problem.ehess is not created.
%   'hess'   -- which corresponds to the default behavior.
% If problem.egrad is already provided and the Hessian is requested, the
% tool builds problem.ehess based on problem.egrad rather than the cost.
% 
% This function requires the following:
%   Matlab version R2021a or later.
%   Deep Learning Toolbox version 14.2 or later.
%
% Support for complex variables in automatic differentation is added in
%   Matlab version R2021b or later.
% There is also better support for Hessian computations in that version.
% Otherwise, see manoptADhelp and complex_example_AD for a workaround, or
% set the 'nohess' flag to tell Manopt not to compute Hessians with AD.
%
% If AD fails for some reasons, the original problem structure 
% is returned with a warning trying to hint at what the issue may be.
% Mostly, issues arise because the manoptAD relies on the Deep Learning
% Toolbox, which itself relies on the dlarray data type, and only a subset
% of Matlab functions support dlarrays:
% 
%   See manoptADhelp for more about limitations and workarounds.
%   See
%   https://ch.mathworks.com/help/deeplearning/ug/list-of-functions-with-dlarray-support.html
%   for an official list of functions that support dlarray.
%
% In particular, sparse matrices are not supported, as well as certain
% standard functions including trace() which can be replaced by ctrace().
%
% There are a few limitations pertaining to specific manifolds.
% For example:
%   fixedrankembeddedfactory: AD creates grad, not egrad; and no Hessian.
%   fixedranktensorembeddedfactory: no AD support.
%   fixedTTrankfactory: no AD support.
%   euclideansparsefactory: no AD support.
%
% Importantly, while AD is convenient and efficient in terms of human time,
% it is not efficient in terms of CPU time: it is expected that AD slows
% down gradient computations by a factor of about 5. Moreover, while AD can
% most often compute Hessians as well, it is often more efficient to
% compute Hessians with finite differences (which is the default in Manopt
% when the Hessian is not provided by the user).
% Thus: it is often the case that
%   problem = manoptAD(problem, 'nohess');
% leads to better overall runtime than
%   problem = manoptAD(problem);
% when calling trustregions(problem).
%
% Some manifold factories in Manopt support GPUs: automatic differentiation
% should work with them too, as usual. See using_gpu_AD for more details.
%
%
% See also: manoptADhelp autograd egradcompute ehesscompute complex_example_AD using_gpu_AD

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

% To do: Add AD to fixedTTrankfactory, fixedranktensorembeddedfactory
% and the product manifold which contains fixedrankembeddedfactory
% or anchoredrotationsfactory

%% Check if AD can be applied to the manifold and the cost function
    
    % Check availability of the Deep Learning Toolbox.
    if ~(exist('dlarray', 'file') == 2)
        error('manopt:AD:dl', ...
        ['It seems the Deep Learning Toolbox is not installed.\n' ...
         'It is needed for automatic differentiation in Manopt.\n' ...
         'If possible, install the latest version of that toolbox and ' ...
         'ideally also Matlab R2021b or later.']);
    end
    
    % Check for a feature of recent versions of the Deep Learning Toolbox.
    if ~(exist('dlaccelerate', 'file') == 2)
        warning('manopt:AD:dlaccelerate', ...
           ['Function dlaccelerate not available:\n If possible, ' ...
            'upgrade to Matlab R2021a or later and use the latest ' ...
            'version of the Deep Learning Toolbox.\n' ...
            'Automatic differentiation may still work but be a lot ' ...
            'slower.\nMoreover, the Hessian is not available in AD.\n' ...
            'Setting flag to ''nohess''. '
            'To disable this warning: ' ...
            'warning(''off'', ''manopt:AD:dlaccelerate'');']);
        flag = 'nohess';
    end

    % The problem structure must provide a manifold and a cost function.
    assert(isfield(problem, 'M') && isfield(problem, 'cost'), ... 
              'The problem structure must contain the fields M and cost.');
    
    % Check the flag value if provided, or set its default value.
    if exist('flag', 'var')
        assert(strcmp(flag, 'nohess') || strcmp(flag, 'hess'), ...
           'The second argument should be either ''nohess'' or ''hess''.');
    else
        flag = 'hess'; % default behavior
    end
    
    % If the gradient and Hessian information is already provided, return.
    if canGetGradient(problem) && canGetHessian(problem)
        warning('manopt:AD:alreadydefined', ...
          ['Gradient and Hessian already defined, skipping AD.\n' ...
           'To disable this warning: ' ...
           'warning(''off'', ''manopt:AD:alreadydefined'');']);
        return;
    end
    
    % Below, it is convenient for several purposes to have a point on the
    % manifold. This makes it possible to investigate its representation.
    x = problem.M.rand();
    
    % AD does not support certain manifolds.
    manifold_name = problem.M.name();
    if contains(manifold_name, 'sparsity')
         error('manopt:AD:sparse', ...
              ['Automatic differentiation currently does not support ' ...
               'sparse matrices, e.g., euclideansparsefactory.']);
    end
    if ( startsWith(manifold_name, 'Product manifold') && ...
        ((sum(isfield(x, {'U', 'S', 'V'})) == 3) && ...
        (contains(manifold_name(), 'rank', 'IgnoreCase', true))) ...
       ) || ( ...
        exist('tenrand', 'file') == 2 && isfield(x, 'X') && ...
        isa(x.X, 'ttensor') ...
       ) || ...
       isa(x, 'TTeMPS')
        error('manopt:AD:fixedrankembedded', ...
             ['Automatic differentiation ' ...
              'does not support fixedranktensorembeddedfactory,\n'...
              'fixedTTrankfactory, and product manifolds containing '...
              'fixedrankembeddedfactory.']);
    end
    
    % complexflag is used to detect if both of the following are true:
    %   A) the problem variables contain complex numbers, and
    %   B) the Matlab version is R2021a or earlier.
    % If so, we attempt a workaround.
    % If Matlab is R2021b or later, then it is not an issue to have
    % complex numbers in the variables.
    complexflag = false;
    % Check if AD can be applied to the cost function by passing the point
    % x we created earlier to problem.cost.
    try
        dlx = mat2dl(x);
        costtestdlx = problem.cost(dlx); %#ok<NASGU>
    catch ME
        % Detect complex number by looking in error message.
        % Note: the error deep:dlarray:ComplexNotSupported is removed 
        % in Matlab R2021b or later
        if (strcmp(ME.identifier, 'deep:dlarray:ComplexNotSupported'))
            try
                % Let's try to run AD with 'complex' workaround.
                dlx = mat2dl_complex(x);
                costtestx = problem.cost(x); %#ok<NASGU>
                costtestdlx = problem.cost(dlx); %#ok<NASGU>
            catch
                error('manopt:AD:complex', ...
                     ['Automatic differentiation failed. ' ...
                      'Problem defining the cost function.\n' ...
                      'Variables contain complex numbers. ' ...
                      'Check your Matlab version and see\n' ...
                      'complex_example_AD.m and manoptADhelp.m for ' ...
                      'help about how to deal with complex variables.']);
            end
            % If no error appears, set complexflag to true.
            complexflag = true;
        else
            % If the error is not related to complex numbers, then the
            % issue is likely with the cost function definition.
            warning('manopt:AD:cost', ...
               ['Automatic differentiation failed. '...
                'Problem defining the cost function.\n'...
                '<a href = "https://www.mathworks.ch/help/deeplearning'...
                '/ug/list-of-functions-with-dlarray-support.html">'...
                'Check the list of functions with AD support.</a>'...
                ' and see manoptADhelp for more information.']);
            return;
        end
    end
    
%% Keep track of what we create with AD
    ADded_gradient = false;
    ADded_hessian  = false;
    
%% Handle special case of fixedrankembeddedfactory first

    % Check if the manifold struct is fixed-rank matrices 
    % with an embedded geometry. For fixedrankembeddedfactory, 
    % only the Riemannian gradient can be computed via AD so far.
    fixedrankflag = false;
    if (sum(isfield(x, {'U', 'S', 'V'})) == 3) && ...
        (contains(manifold_name, 'rank', 'IgnoreCase', true)) && ...
        (~startsWith(manifold_name, 'Product manifold'))
    
        if ~strcmp(flag, 'nohess')
            warning('manopt:AD:fixedrank', ...
              ['Computating the exact Hessian via AD is not supported ' ...
               'for fixedrankembeddedfactory.\n' ...
               'Setting flag to ''nohess''.\nTo disable this warning: ' ...
               'warning(''off'', ''manopt:AD:fixedrank'');']);
            flag = 'nohess';
        end
        
        % Set the fixedrankflag to true to prepare for autgrad.
        fixedrankflag = true;
        % If no gradient information is provided, compute grad using AD.
        % Note that here we define the Riemannian gradient.
        if ~canGetGradient(problem)
            problem.autogradfunc = autograd(problem, fixedrankflag);
            problem.grad = @(x) gradcomputefixedrankembedded(problem, x);
            problem.costgrad = @(x) costgradcomputefixedrankembedded(problem, x);
            ADded_gradient = true;
        end
        
    end
    
%% Compute the euclidean gradient and the euclidean Hessian via AD
    
    % Provide egrad and (if requested) ehess via AD.
    % Manopt converts to Riemannian derivatives via egrad2rgrad and
    % ehess2rhess as usual: no need to worry about this here.
    if ~fixedrankflag
        
        if ~canGetGradient(problem)
            problem.autogradfunc = autograd(problem);
            problem.egrad = @(x) egradcompute(problem, x, complexflag);
            problem.costgrad = @(x) costgradcompute(problem, x, complexflag);
            ADded_gradient = true;
        end
        
        if ~canGetHessian(problem) && strcmp(flag, 'hess')
            problem.ehess = @(x, xdot, store) ...
                                     ehesscompute(problem, x, xdot, ...
                                                  store, complexflag);
            ADded_hessian = true;
        end
        
    end
            
    
%% Check whether the gradient / Hessian we AD'ded actually work.

    % Some functions are not supported to be differentiated with AD in the
    % Deep Learning Toolbox, e.g., cat(3, A, B).
    % In this clean-up phase, we check if things actually work, and we
    % remove functions if they do not, with a warning.
    
    if ADded_gradient && ~fixedrankflag
        
        try 
            egrad = problem.egrad(x);
        catch
            warning('manopt:AD:failgrad', ...
               ['Automatic differentiation for gradient failed. '...
                'Problem defining the cost function.\n'...
                '<a href = "https://www.mathworks.ch/help/deeplearning'...
                '/ug/list-of-functions-with-dlarray-support.html">'...
                'Check the list of functions with AD support.</a>'...
                ' and see manoptADhelp for more information.']);
            problem = rmfield(problem, 'autogradfunc');
            problem = rmfield(problem, 'egrad');
            problem = rmfield(problem, 'costgrad');
            if ADded_hessian
                problem = rmfield(problem, 'ehess');
            end
            return;
        end
        
        if isNaNgeneral(egrad)
            warning('manopt:AD:NaN', ...
                   ['Automatic differentiation for gradient failed. '...
                    'Problem defining the cost function.\n'...
                    'NaN comes up in the computation of egrad via AD.\n'...
                    'Check the example thomson_problem.m for help.']);
            problem = rmfield(problem, 'autogradfunc');
            problem = rmfield(problem, 'egrad');
            problem = rmfield(problem, 'costgrad');
            if ADded_hessian
               problem = rmfield(problem, 'ehess');
            end
            return;
        end
        
    end
        
    
    if ADded_hessian
        
        % Randomly generate a vector in the tangent space at x.
        xdot = problem.M.randvec(x);
        store = struct();
        try 
            ehess = problem.ehess(x, xdot, store);
        catch
            warning('manopt:AD:failhess', ...
                   ['Automatic differentiation for Hessian failed. ' ...
                    'Problem defining the cost function.\n' ...
                    '<a href = "https://www.mathworks.ch/help/deeplearning' ...
                    '/ug/list-of-functions-with-dlarray-support.html">' ...
                    'Check the list of functions with AD support.</a>' ...
                    ' and see manoptADhelp for more information.']);
            problem = rmfield(problem, 'ehess');
            return;
        end
        
        if isNaNgeneral(ehess)
            warning('manopt:AD:NaN', ...
                   ['Automatic differentiation for Hessian failed. ' ...
                    'Problem defining the cost function.\n' ...
                    'NaN comes up in the computation of egrad via AD.\n' ...
                    'Check the example thomson_problem.m for help.']);
            problem = rmfield(problem, 'ehess');
            return;
        end
        
    end
        
    % Check the case of fixed-rank matrices as embedded submanifold.
    if ADded_gradient && fixedrankflag
        try 
            grad = problem.grad(x);
        catch
            warning('manopt:AD:costfixedrank', ...
                   ['Automatic differentiation for gradient failed. ' ...
                    'Problem defining the cost function.\n' ...
                    '<a href = "https://www.mathworks.ch/help/deeplearning' ...
                    '/ug/list-of-functions-with-dlarray-support.html">' ...
                    'Check the list of functions with AD support.</a>' ...
                    ' and see manoptADhelp for more information.']);
            problem = rmfield(problem, 'autogradfunc');                
            problem = rmfield(problem, 'grad');
            problem = rmfield(problem, 'costgrad');
            return;
        end
        
        if isNaNgeneral(grad)
            warning('manopt:AD:NaN', ...
                   ['Automatic differentiation for gradient failed. ' ...
                    'Problem defining the cost function.\n' ...
                    'NaN comes up in the computation of grad via AD.\n' ...
                    'Check the example thomson_problem.m for help.']);
            problem = rmfield(problem, 'autogradfunc');
            problem = rmfield(problem, 'grad');
            problem = rmfield(problem, 'costgrad');
            return;
        end
        
    end
    
    
end
