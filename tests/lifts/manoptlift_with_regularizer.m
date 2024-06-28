function [upstairs, downstairs] = manoptlift_with_regularizer(downstairs, lift, ADflag)
% Manopt tool to lift an optimization problem through a parameterization.
%
%  QUICKLY HACKED TOGETHER TO TEST SOMETHING -- documentation isn't right.
%
% function [upstairs, downstairs] = manoptlift_with_regularizer(downstairs, lift)
% function [upstairs, downstairs] = manoptlift_with_regularizer(downstairs, lift, 'AD')
% function [upstairs, downstairs] = manoptlift_with_regularizer(downstairs, lift, 'noAD')
% function [upstairs, downstairs] = manoptlift_with_regularizer(downstairs, lift, 'ADnohess')
%
% Given an optimization problem (downstairs) on a manifold (lift.N),
% produces a new optimization problem (upstairs) on a manifold (lift.M) by
% composing the cost function downstairs with the map lift.phi.
%
% Inputs:
%   downstairs is a Manopt problem structure
%   lift is a Manopt lift structure
%   ADflag (optional) is a string
% 
% Outputs:
%   upstairs is a Manopt problem structure
%   downstairs is a Manopt problem structure (possibly modified from input)
%
% The lift structure contains at least the following fields:
%   lift.M is a manifold structure (obtained from a factory)
%   lift.N is a manifold structure as well
%   lift.phi is a map from M to N
%
% Ideally, lift also contains the following additional fields:
%   lift.Dphi is the differential of phi: Dphi(y, v) = Dphi(y)[v]
%   lift.Dphit is the adjoint of Dphi: Dphit(y, u) = Dphi(y)*[u]
%   lift.hesshw, if N is linear, is the Hessian of a lifted linear map,
%                as follows: given w downstairs, let h(y) = <phi(y), w>;
%                then, hesshw(y, v, w) = Hess h(y)[v].
%                More generally, it is the Hessian of h = q o phi, where
%                q:N->R satisfies grad q(x) = w (for the fixed x = phi(y)).
%   lift.hessqw is assumed to be a zero map if omitted. If it is not
%               omitted, then hessqw(x, u, w) is the Hessian of q at x
%               along u, where q is the map chosen in defining hesshw.
%   lift.embedded is a boolean:
%       false (default) means Dphi, Dphit, hesshw are defined for phi:M->N.
%       true means they are defined for phi:E->N where E is the embedding
%                       space of M.
%
% The manifold lift.N must be the same as that of downstairs.M.
% If downstairs.M is not defined, then it is set to be lift.N.
% The manifold of upstairs.M is set to be lift.M.
%
% If f : N -> R is the cost function over lift.N defined in downstairs,
% then g : M -> R is the cost function over lift.M defined in upstairs,
% via the composition g = f o phi.
%
% ADflag can be omitted, or it can be set to one of the following strings:
%   'AD' to run manoptAD (automatic differentiation) on downstairs,
%   'ADnohess' to do the latter for the gradient but not the Hessian,
%   'noAD' or '' to not run automatic differentiation (default).
%
%
% Heads up: at this time, lift composition may not work because of caching.
%
%
% This tools is based on general theory for reparameterization of nonconvex
% optimization problems developed in:
%
%   Levin, Kileel, Boumal
%   The effect of smooth parametrizations on nonconvex optimization landscapes
%   Mathematical Programming, 2024
%   https://link.springer.com/article/10.1007/s10107-024-02058-3
%   https://arxiv.org/abs/2207.03512
%
%
% See also: hadamardlift burermonteirolift burermonteiroLRlift
%           cubeslift ballslift

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 18, 2024.
% Contributors: 
% Change log: 


    % To consider:
    %
    % Should lift functions have access to the store?
    % Or should manoptlift make it so that the image lift.phi(x) is
    % cached by default? And should this be controlled by an option?
    %
    % Implement a tool productlift?
    % Have a generic identity lift buil-in so that we could take a product
    % between a lift and (in some sense) a manifold?
    % Should we include phiinverse for diffeomorphism lifts?
    %
    % Composition: caching is potentially not safe in its current form if
    % we want to compose lifts (that is, apply manoptlift to a problem,
    % then again on the resulting problem).
    % Perhaps the field name 'grad_downstairs__' could be tagged with some
    % unique identified corresponding to this each call to manoptlift
    % (e.g., with a global counter that's incremented each time). Would the
    % other cached fields ('grad__' and 'egrad__') be safe?


    assert(isstruct(lift) && ...
           isfield(lift, 'phi') && ...
           isfield(lift, 'M') && ...
           isfield(lift, 'N'), ...
           'The lift must be a structure with certain mandatory fields.');

    % By default, a lift is to the manifold, not its embedding.
    if ~isfield(lift, 'embedded')
        lift.embedded = false;
    end

    % If the manifold is not specified in the downstairs problem, assume it
    % is the right one, namely, the codomain of the lift.
    if ~isfield(downstairs, 'M')
        downstairs.M = lift.N;
    end

    % The domain of the downstairs problem should match the codomain of the
    % lift.
    if ~strcmp(lift.N.name(), downstairs.M.name())
        warning('manoptlift:domainclash', ...
                ['The lift''s image name is: %s\n' ...
                 'The downstairs problem''s domain is: %s\n' ...
                 'They should be the same, but they are not.'], ...
                lift.N.name(), downstairs.M.name());
    end

    % The linesearch field (if present) is ignored: just let the user know.
    if isfield(downstairs, 'linesearch')
        warning('manoptlift:linesearch', ...
                ['The field downstairs.linesearch is ignored ' ...
                 'when constructing the upstairs problem.']);
    end

    % Determine if the caller wants us to run Automatic Differentiation on
    % the downstairs problem before we proceed with lifting it.
    if ~exist('ADflag', 'var') || isempty(ADflag)
        ADflag = 'noAD';
        callAD = false;
    end
    switch ADflag
        case {'AD'}
            ADflag = 'hess';
            callAD = true;
        case {'ADnohess'}
            ADflag = 'nohess';
            callAD = true;
        case {'noAD', ''}
            callAD = false;
        otherwise
            warning(['ADflag must be empty, omitted, or one of these:\n' ...
                     '''AD'', ''ADnohess'', ''noAD''.']);
    end
    if callAD
        % Force AD to run on a point that can be reached through the lift.
        % This specificity is the only reason why we have some AD logic in
        % this tool. Otherwise, we could just let the caller run manoptAD
        % on their problem (in one line) before running this tool. For most
        % problems that would work, but the format here is more robust.
        y = lift.M.rand();
        downstairs = manoptAD(downstairs, ADflag, lift.phi(y));
    end

    % Start creating the upstairs problem structure
    upstairs.M = lift.M;

    if canGetCost(downstairs)

        upstairs.cost = @cost;

    end

    if canGetGradient(downstairs) && ...
       isfield(lift, 'Dphit')

        if lift.embedded
            upstairs.egrad = @gradient;
        else
            upstairs.grad = @gradient;
        end

    end
    
    if canGetGradient(downstairs) && ...
       canGetHessian(downstairs) && ...
       isfield(lift, 'Dphi') && ...
       isfield(lift, 'Dphit') && ...
       isfield(lift, 'hesshw')
    
        if lift.embedded
            upstairs.ehess = @hessian;
        else
            upstairs.hess = @hessian;
        end

    end

    if canGetCost(downstairs) && ...
       canGetGradient(downstairs) && ...
       isfield(lift, 'Dphit')

        upstairs.costgrad = @costgrad;

    end



    function val = cost(y, storedb, key)

        % Map the point from upstairs (y) to downstairs (x).
        x = lift.phi(y);

        % The call to getCost computes and caches the cost function value.
        % Notice how we simply forward the storedb and key from the
        % upstairs problem to the downstairs problem: see comments below.
        val = getCost(downstairs, x, storedb, key);

        val = val + lift.rho(y);

    end

    % ! storedb is associated to the upstairs problem.
    %   key identifies the point y, matching it with a store.
    %   From y, we get x = phi(y) unambiguously.
    %   We also use the store of y to cache information about x.
    %   Therefore, we must be careful with Manopt's automatic caching:
    %   there can be no ambiguity that quantities cached by the getXYZ
    %   tools must refer to y and the problem upstairs. This can be
    %   compromised by calls of the form
    %      getXYZ(downstairs, x, storedb, key)
    %   for which automatic caching may store quantities pertaining to the
    %   downstairs problem at x in the store of y. We must overrule these.
    %   Manopt automatically caches:
    %   * cost__  -> ok since cost upstairs at y == cost downstairs at x.
    %   * grad__  -> not ok since upstairs and downstairs gradients differ. 
    %   * egrad__ -> idem.
    %   The logic developed below is that if a gradient of any kind is
    %   queried upstairs, then that will necessarily require computing the
    %   gradient downstairs. We make sure to cache the latter in a new
    %   field called grad_downstairs__.

    function grad_downstairs = get_gradient_downstairs(x, storedb, key)
        store = storedb.get(key);
        if ~isfield(store, 'grad_downstairs__')
            % If the gradient (Riemannian or embedded) was computed, then
            % necessarily that involved computing the gradient downstairs.
            % All of that should have been cached in the store. Thus, if
            % grad_downstairs__ is inexistent, it (should) mean that no
            % gradient upstairs was computed at y, and therefore that none
            % was cached either. This is important, because if it is cached
            % then that could interfere with the call to getGradient below.
            store_changed = false;
            if isfield(store, 'grad__')
                store = rmfield(store, 'grad__');
                store_changed = true;
            end
            if isfield(store, 'egrad__')
                store = rmfield(store, 'egrad__');
                store_changed = true;
            end
            if store_changed
                % Save the modified store now, before calling getGradient. 
                storedb.set(store, key);
                warning('manoptlift:wrongcache', ...
                        ['This should not happen. ' ...
                         'Please let us know on the forum.']);
            end
            store.grad_downstairs__ = getGradient(downstairs, x, ...
                                                             storedb, key);
            storedb.set(store, key);
        end
        grad_downstairs = store.grad_downstairs__;
    end

    function grad = gradient(y, storedb, key)

        x = lift.phi(y);

        grad_downstairs = get_gradient_downstairs(x, storedb, key);

        % This will be cached by the caller in store.grad__ or
        % store.egrad__ as appropriate (determined by lift.embedded).
        grad = lift.Dphit(y, grad_downstairs);

        grad = lift.M.lincomb(y, 1, grad, 1, lift.gradrho(y));

    end

    function hess = hessian(y, v, storedb, key)

        x = lift.phi(y);
        u = lift.Dphi(y, v);

        grad_downstairs = get_gradient_downstairs(x, storedb, key);

        hess_downstairs = getHessian(downstairs, x, u, storedb, key);

        if isfield(lift, 'hessqw')
            hess_downstairs = ...
                  lift.N.lincomb(x, 1, hess_downstairs, ...
                                   -1, lift.hessqw(x, u, grad_downstairs));
        end

        hess = lift.M.lincomb(y, ...
               1, lift.Dphit(y, hess_downstairs), ...
               1, lift.hesshw(y, v, grad_downstairs));

        hess = lift.M.lincomb(y, 1, hess, 1, lift.hessrho(y, v));
        
    end

    function [val, grad] = costgrad(y, storedb, key)

        % A call to getCost, getGradient or getCostGradient finds its way
        % through to this code only if the cost and/or the (upstairs)
        % gradient were not already cached.

        x = lift.phi(y);

        % Do the actual work.
        switch nargout
            case 1
                % This call caches the cost function value.
                val = getCost(downstairs, x, storedb, key);

                val = val + lift.rho(y);

            case 2

                % The call to getCostGrad includes logic to avoid
                % recomputing the cost if it was already known for this y.
                % It caches the cost function value and the gradient.
                [val, grad_downstairs] = getCostGrad(downstairs, x, ...
                                                             storedb, key);
                grad = lift.Dphit(y, grad_downstairs);

                if lift.embedded
                    grad = lift.M.egrad2rgrad(y, grad);
                end

                val = val + lift.rho(y);
                grad = lift.M.lincomb(y, 1, grad, 1, lift.gradrho(y));

                % ! The call to getCostGrad cached val and grad_downstairs
                %   in store.cost__ and store.grad__.
                %   Once the call to the current function terminates,
                %   the calling function (likely also getCostGrad or
                %   getGradient) will overwrite grad__ with the output of
                %   this function, that is, grad: this is good.
                %   However, we also want to cache grad_downstairs, so we
                %   do it manually now:
                store = storedb.get(key);
                store.grad_downstairs__ = grad_downstairs;
                storedb.set(store, key);

            otherwise
                error('costgrad can have 1 or 2 outputs.');
        end

    end

end
