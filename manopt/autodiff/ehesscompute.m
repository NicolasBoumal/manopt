function [ehess,store] = ehesscompute(problem, x, xdot, store, complexflag)
% Computes the Euclidean Hessian of the cost function at x along xdot via AD.
%
% function [ehess, store] = ehesscompute(problem, x, xdot)
% function [ehess, store] = ehesscompute(problem, x, xdot, store)
% function [ehess, store] = ehesscompute(problem, x, xdot, store, complexflag)
%
% This file requires Matlab R2021a or later.
%
% Returns the Euclidean Hessian of the cost function described in the
% problem structure at the point x along xdot. Returns store structure 
% which stores the Euclidean gradient and AD trace in order to avoid
% redundant computation of hessian by-vector product at the same point x.
%
% complexflag is bool variable which indicates whether or not the cost  
% function and the manifold described in the problem structure involves 
% complex numbers and meanwhile the Matlab version is R2021a or earlier.
%
% Note: the Euclidean hessian by-vector product is computed through
% differentiating the inner product between egrad and xdot, thus the 
% result is valid only when second-order partial derivatives commute. 
% When the egrad function has already been specified by the user, the
% euclidean gradient is computed according to the egrad and otherwise 
% according to the cost function.
%
% See also: manoptAD mat2dl dl2mat dl2mat_complex mat2dl_complex innerprodgeneral cinnerprodgeneral

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:     

    %% Prepare Euclidean gradient
   
    % check availability 
    assert(isfield(problem,'M') && isfield(problem,'cost'),...,
    'problem structure must contain fields M and cost.');
    assert(exist('dlarray', 'file') == 2, ['Deep learning tool box is '... 
    'needed for automatic differentiation']);
    assert(exist('dlaccelerate', 'file') == 2, ['AD failed when computing'...
        'the hessian. Please upgrade to Matlab R2021a or later.'])

    % check whether the user has specified the egrad already
    egradflag = false;
    if isfield(problem,'egrad') && ~isfield(problem,'autogradfunc')
        egradflag = true;
    end

    % check the Matlab version and the complex number
    if ~exist('complexflag','var')
        complexflag = false;
    end
    % obtain cost funtion via problem
    costfunction = problem.cost;
    
    % prepare euclidean gradient if not yet
    if (~exist('store','var') || ~isfield(store,'dlegrad')) 
        
        % create a tape and start recording the trace that records the 
        % computation of the Euclidean gradient. the destruction of record 
        % object cleans up the tape, which is done at the same time when 
        % the store is renewed after each iteration,
        tm = deep.internal.recording.TapeManager();
        record = deep.internal.startTracingAndSetupCleanup(tm);
        
        % compute the euclidean gradient of the cost function at x
        [dlx,dlegrad] = subautograd(costfunction,complexflag,x);
        
        % store the trace, euclidean gradient and the point dlx
        store.dlegrad = dlegrad;
        store.dlx = dlx;
        store.tm = tm;
        store.record = record;
       
    end
    
    % define gradient computation function which is similar to autograd
    function [dlx,dlegrad] = subautograd(costfunction,complexflag,x)
        
        % convert x into dlarrays to prepare for AD
        if complexflag == true
            dlx = mat2dl_complex(x);
        else
            dlx = mat2dl(x);
        end
        
        % convert dlx into recording arrays
        dlx = deep.internal.recording.recordContainer(dlx);
        
        % if the user has defined the egrad, compute the Euclidean gradient 
        % and keep the trace
        if egradflag
            try 
                dlegrad = problem.egrad(dlx);
            catch
                egradflag = false;
            end
        end

        % otherwise, compute the Euclidean gradient from the cost function
        if ~egradflag
            y = costfunction(dlx);
            % in case that the user forgot to take the real part of the cost
            % when dealing with complex problems and meanwhile the Matlab
            % version is R2021a or earlier, take the real part for AD
            if iscstruct(y)
                y = creal(y);
            end
            % call dlgradient to compute the Euclidean gradient
            % trace the backward pass in order to compute higher order
            % derivatives in the further steps 
            dlegrad = dlgradient(y,dlx,'RetainData',true,'EnableHigherDerivatives',true);
        end
    end
    
    %% compute the Euclidean Hessian of the cost function at x along xdot
    
    % prepare ingredients 
    tm = store.tm; %#ok<NASGU>
    record = store.record; %#ok<NASGU>
    dlegrad = store.dlegrad;
    dlx = store.dlx;
    
    % To compute Euclidean Hessian vector product, rotations manifold, 
    % unitary manifold and essential manifold requires first converting 
    % the representation of the tangent vector into the ambient space. 
    % if the problem is a product manifold, in addition to the above
    % manifolds, the xdot of the other manifolds remain the same
    if contains(problem.M.name(),'Rotations manifold SO','IgnoreCase',true)..., 
            ||  contains(problem.M.name(),'Unitary manifold','IgnoreCase',true)...,
            || (contains(problem.M.name(),'Product rotations manifold','IgnoreCase',true) &&..., 
            contains(problem.M.name(),'anchors'))...,
            || contains(problem.M.name(),'essential','IgnoreCase',true)
        xdot = problem.M.tangent2ambient(x, xdot);
    end 
    
    % compute the inner product between the Euclidean gradient and xdot
    if complexflag == true
        z = cinnerprodgeneral(dlegrad, xdot);
    else
        z = innerprodgeneral(dlegrad, xdot);
    end
    
    % compute derivatives of the inner product w.r.t. dlx
    ehess = dlgradient(z,dlx,'RetainData',false,'EnableHigherDerivatives',false);
    
    % obtain the numerical representation 
    if complexflag == true
        ehess = dl2mat_complex(ehess);
    else
        ehess = dl2mat(ehess);
    end
    
    
    % in case that the user is optimizing over anchoredrotationsfactory
    % ehess of anchors with indices in A should be zero
    if (contains(problem.M.name(),'Product rotations manifold') &&..., 
            contains(problem.M.name(),'anchors'))
        A = findA_anchors(problem);
        ehess(:, :, A) = 0;
    end
    
end
