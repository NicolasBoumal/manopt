function problem = manoptAD(problem, varargin) 
% Preprocess automatic differentiation for the problem structure
%
% function problem = manoptAD(problem)
% function problem = manoptAD(problem, 'egrad')
% function problem = manoptAD(problem, 'ehess')
%
% Check if the automatic differentiation provided in the deep learning tool
% box can be applied to computing the euclidean gradient and the euclidean
% hessian given the manifold and the cost function described in the problem
% structure. If AD fails for some reasons, the original problem structure 
% is returned and the approx. of gradient or hessian will then be used 
% as usual. Otherwise, the problem structure with additional fields: 
% egrad, costgrad and ehess is returned. If the user only wants the  
% gradient or the hessian information, the second argument 'egrad' or  
% 'ehess' should be specified. If the egrad or the ehess is alrealdy 
% provided by the user, the complement information is returned by calling 
% manoptAD(problem). e.g. if the user has already provided the gradient 
% information, he can call problem = manoptAD(problem,'ehess') or 
% problem = manoptAD(problem) to obtain the ehess via AD. In this
% case, the ehess is computed according to the egrad instead of the cost.
%
% In the case that the manifold is the set of fixed-rank matrices with 
% an embedded geometry, it is more efficient to compute the Riemannian 
% gradient directly. However, computing the exact Riemannian Hessian by 
% vector product via AD is currently not supported. By calling 
% manoptAD, the problem struct with additional fields grad and costgrad
% is returned. Besides, optimizing on fixedranktensorembeddedfactory and 
% fixedTTrankfactory via AD is currently not supported.
%
% Note: The current functionality of AD relies on Matlab's deep learning
% tool box, which has the inconvenient effect that we cannot control the
% limitations. Firstly, AD does not support sparse matrices so far. Try 
% converting sparse arrays into full arrays in the cost function. Secondly,
% math operations involving complex numbers between dlarrays is not 
% supported for Matlab R2021a or earlier. To fully exploit the convenience 
% of AD, please update to Matlab R2021b or later if possible. If the user 
% cannot have access to Matlab R2021b or later, manopt provides an 
% alternative way to deal with complex problems. see complex_example_AD 
% and manoptADhelp for more information. Thirdly, check the list of functions
% with AD support when defining the cost function. See the official website
% https://ch.mathworks.com/help/deeplearning/ug/list-of-functions-with-dlarray-support.html
% and manoptADhelp for more information. To run AD on GPU, set gpuflag = true 
% in the problem structure and store related arrays on GPU as usual. 
% See using_gpu_AD for more details.
%
% See also: autograd, egradcompute, ehesscompute, complex_example_AD
% manoptADhelp, using_gpu_AD

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 
%
% To do: Add AD to fixedTTrankfactory, fixedranktensorembeddedfactory
% and the product manifold which contains fixedrankembeddedfactory
% or anchoredrotationsfactory

%% Check if AD can be applied to the manifold and the cost function
    
    assert(isfield(problem,'M') && isfield(problem,'cost'), ... 
              'the problem structure must contain the fields M and cost.');
    if nargin==2 
        assert(strcmp(varargin,'egrad')|| strcmp(varargin,'ehess'), ...
            'the second argument should be either ''egrad'' or ''ehess''');
    end
    % if the gradient and hessian information is provided already, return
    if  (isfield(problem,'egrad') && isfield(problem,'ehess'))...
            || (isfield(problem,'egrad') && isfield(problem,'hess'))...
            || (isfield(problem,'grad') && isfield(problem,'ehess'))...
            || (isfield(problem,'grad') && isfield(problem,'hess'))...
            || (isfield(problem,'costgrad') && isfield(problem,'ehess'))...
            || (isfield(problem,'costgrad') && isfield(problem,'hess'))
        return
    % AD does not support euclideansparsefactory so far.
    elseif contains(problem.M.name(),'sparsity')
         warning('manopt:sparse',['Automatic differentiation currently does not support '...
                    'sparse matrices']);
        return
    % check availability.
    elseif ~(exist('dlarray', 'file') == 2)
        warning('manopt:dl',['It seems the Deep learning tool box is not installed.'...
         '\nIt is needed for automatic differentiation.\nPlease install the '...
         'latest version of the deep learning tool box and \nupgrade to Matlab '...
         'R2021b or later if possible.'])
        return
    else 
        % complexflag is used to detect if the problem defined contains
        % complex numbers and meanwhile the Matlab version is R2021a or earlier.
        complexflag = false;
        % check if AD can be applied to the cost function by passing a
        % point on the manifold to problem.cost.
        x = problem.M.rand();
        problem_name = problem.M.name();
        % check fixed-rank exceptions
        if  (startsWith(problem_name,'Product manifold') &&...
            ((sum(isfield(x,{'U','S','V'}))==3) &&...
        (contains(problem_name(),'rank','IgnoreCase',true)))) || ...
        (exist('tenrand', 'file')==2 && isfield(x,'X') && ...
        isa(x.X,'ttensor')) || isa(x,'TTeMPS')
            warning('manopt:AD:fixedrankembedded',['Automatic differentiation ' ...
                ' currently does not support fixedranktensorembeddedfactory,\n'...
                'fixedTTrankfactory, and product manifolds containing '...
                'fixedrankembeddedfactory.']);           
            return
        end
        try
            dlx = mat2dl(x);
            costtestdlx = problem.cost(dlx); %#ok<NASGU>
        catch ME
            % detect complex number by looking up error message
            % Note: the error deep:dlarray:ComplexNotSupported is removed 
            % in Matlab R2021b or later
            if (strcmp(ME.identifier,'deep:dlarray:ComplexNotSupported'))
                try
                    dlx = mat2dl_complex(x);
                    costtestx = problem.cost(x); %#ok<NASGU>
                    costtestdlx = problem.cost(dlx); %#ok<NASGU>
                catch
                    warning('manopt:complex', ...
                    ['Automatic differentiation failed. ' ...
                     'Problem defining the cost function.\n' ...
                     'Variables contain complex numbers. ' ...
                     'Check your Matlab version and see\n' ...
                     'complex_example_AD.m and manoptADhelp.m for help ' ...
                     'about how to deal with complex variables.']);
                    return
                end
                % if no error appears, set complexflag to true
                complexflag = true;
            else
                % if the error is not related to complex number, then it
                % must be the problem of defining the cost function
                warning('manopt:costAD',['Automatic differentiation failed. '...
                    'Problem defining the cost function.\n'...
                    '<a href = "https://www.mathworks.ch/help/deeplearning'...
                    '/ug/list-of-functions-with-dlarray-support.html">'...
                    'Check the list of functions with AD support.</a>'...
                    ' and see manoptADhelp for more information.']);
                return   
            end
        end                   
    end
    if ~(exist('dlaccelerate', 'file') == 2)
        warning('manopt:dlaccelerate', ...
            ['Function dlaccelerate is not available:\nPlease ' ...
            'upgrade to Matlab 2021a or later and the latest deep\nlearning ' ...
            'toolbox version if possible.\nMeanwhile, auto-diff ' ...
            'may be somewhat slower.\nThe hessian is not available as well.\n' ...
            'To disable this warning: warning(''off'', ''manopt:dlaccelerate'')']);
    end
%% compute the euclidean gradient and the euclidean hessian via AD

    % check if the manifold struct is fixed-rank matrices 
    % with an embedded geometry. for fixedrankembedded factory, 
    % only the Riemannian gradient can be computed via AD so far.
    fixedrankflag = 0;
    if (sum(isfield(x,{'U','S','V'}))==3) &&...
        (contains(problem_name,'rank','IgnoreCase',true)) &&...
        (~startsWith(problem_name,'Product manifold'))
        if ~(nargin==2 && strcmp(varargin,'egrad'))
            warning('manopt:fixedrankAD',['Computating the exact hessian via '...
            'AD is currently not supported.\n'...
            'To disable this warning: warning(''off'', ''manopt:fixedrankAD'')']);
        end
        % set the fixedrankflag to 1 to prepare for autgrad
        fixedrankflag = 1;
        % if no gradient information is provided, compute grad using AD
        if ~isfield(problem,'egrad') && ~isfield(problem,'grad')...
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
    hessianflag = false;
    switch nargin
        case 1
    % if only the hessian information is provided, compute egrad 
    % hessianflag indicates whether or not ehess or hess has provided already 
        if ~isfield(problem,'egrad') && ~isfield(problem,'grad')...
            && ~isfield(problem,'costgrad') && (isfield(problem,'ehess')...
            || isfield(problem,'hess'))
        
            problem.autogradfunc = autograd(problem);
            problem.egrad = @(x) egradcompute(problem,x,complexflag);
            problem.costgrad = @(x) costgradcompute(problem,x,complexflag);
            hessianflag = true;
        
    % if only the gradient information is provided, compute ehess     
        elseif ~isfield(problem,'ehess') && ~isfield(problem,'hess')...
            && (isfield(problem,'costgrad') || isfield(problem,'grad')...
            || isfield(problem,'egrad')) && (fixedrankflag == 0) &&...
            (exist('dlaccelerate', 'file') == 2)

            problem.ehess = @(x,xdot,store) ehesscompute(problem,x,xdot,store,complexflag);
        
    % otherwise compute both egrad and ehess via automatic differentiation      
        elseif fixedrankflag == 0
            problem.autogradfunc = autograd(problem);
            problem.egrad = @(x) egradcompute(problem,x,complexflag);
            problem.costgrad = @(x) costgradcompute(problem,x,complexflag);
            if exist('dlaccelerate', 'file') == 2
                problem.ehess = @(x,xdot,store) ehesscompute(problem,x,xdot,store,complexflag);
            end
        end
        
        case 2
    % provide the relevant fields according to varargin
            if strcmp(varargin,'egrad')
                problem.autogradfunc = autograd(problem);
                problem.egrad = @(x) egradcompute(problem,x,complexflag);
                problem.costgrad = @(x) costgradcompute(problem,x,complexflag);
                hessianflag = true;
            elseif strcmp(varargin,'ehess') && (exist('dlaccelerate', 'file') == 2)
                problem.ehess = @(x,xdot,store) ehesscompute(problem,x,xdot,store,complexflag);
            end
            
        otherwise
            error('Too many input arguments');

    end
            
    
%% check whether the cost function can be differentiated or not

    % some functions are not supported to be differentiated with AD in the
    % deep learning tool box. e.g.cat(3,A,B). Check availablility of egrad,
    % if not, remove relevant fields such as egrad and ehess.
    
    if isfield(problem,'autogradfunc') && (fixedrankflag == 0)
        try 
            egrad = problem.egrad(x);
        catch
                warning('manopt:costAD',['Automatic differentiation failed. '...
                    'Problem defining the cost function.\n'...
                    '<a href = "https://www.mathworks.ch/help/deeplearning'...
                    '/ug/list-of-functions-with-dlarray-support.html">'...
                    'Check the list of functions with AD support.</a>'...
                    'and see manoptADhelp for more information.']);
                problem = rmfield(problem,'autogradfunc');
                problem = rmfield(problem,'egrad');
                problem = rmfield(problem,'costgrad');
            if ~hessianflag && (exist('dlaccelerate', 'file') == 2)
                problem = rmfield(problem,'ehess');
            end
            return
        end
        if isNaNgeneral(egrad)
            warning('manopt:NaNAD',['Automatic differentiation failed. '...
                    'Problem defining the cost function.\n'...
                    'NaN comes up in the computation of egrad via AD.\n'...
                    'Check the example thomson_problem.m for more information.']);
            problem = rmfield(problem,'autogradfunc');
            problem = rmfield(problem,'egrad');
            problem = rmfield(problem,'costgrad');
            if ~hessianflag && (exist('dlaccelerate', 'file') == 2)
               problem = rmfield(problem,'ehess');
            end
            return
        end
    % if only the egrad or grad is provided, check ehess
    elseif ~isfield(problem,'autogradfunc') && (fixedrankflag == 0) &&...
            ~hessianflag && isfield(problem,'ehess')
        % randomly generate a point in the tangent space at x
        xdot = problem.M.randvec(x);
        store = struct();
        try 
            ehess = problem.ehess(x,xdot,store);
        catch
            warning('manopt:costAD',['Automatic differentiation failed. '...
                    'Problem defining the cost function.\n'...
                    '<a href = "https://www.mathworks.ch/help/deeplearning'...
                    '/ug/list-of-functions-with-dlarray-support.html">'...
                    'Check the list of functions with AD support.</a>'...
                    'and see manoptADhelp for more information.']);
            problem = rmfield(problem,'ehess');
            return
        end
        if isNaNgeneral(ehess)
            warning('manopt:NaNAD',['Automatic differentiation failed. '...
                    'Problem defining the cost function.\n'...
                    'NaN comes up in the computation of egrad via AD.\n'...
                    'Check the example thomson_problem.m for more information.']);
            problem = rmfield(problem,'ehess');
            return
        end
    % check the case of fixed rank matrices endowed with an embedded geometry 
    elseif isfield(problem,'autogradfunc') && fixedrankflag == 1
        try 
            grad = problem.grad(x);
        catch
            warning('manopt:costAD',['Automatic differentiation failed. '...
                    'Problem defining the cost function.\n'...
                    '<a href = "https://www.mathworks.ch/help/deeplearning'...
                    '/ug/list-of-functions-with-dlarray-support.html">'...
                    'Check the list of functions with AD support.</a>'...
                    'and see manoptADhelp for more information.']);
            problem = rmfield(problem,'autogradfunc');                
            problem = rmfield(problem,'grad');
            problem = rmfield(problem,'costgrad');
            return
        end
        if isNaNgeneral(grad)
            warning('manopt:NaNAD',['Automatic differentiation failed. '...
                    'Problem defining the cost function.\n'...
                    'NaN comes up in the computation of egrad via AD.\n'...
                    'Check the example thomson_problem.m for more information.']);
            problem = rmfield(problem,'autogradfunc');
            problem = rmfield(problem,'grad');
            problem = rmfield(problem,'costgrad');
            return
        end
        
    end
    
    
end
