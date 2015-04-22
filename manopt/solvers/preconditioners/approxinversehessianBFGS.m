function hessfun =  approxinversehessianBFGS(problem, options)
% Hessian approx. fnctn handle based on inv. Hessian approx with BFGS.
%
% function hessfun = approxinversehessianBFGS(problem)
% function hessfun = approxinversehessianBFGS(problem, options)
%
% Input:
%
% A Manopt problem structure (already containing the manifold and enough
% information to compute the cost gradient) and an options structure
% (optional), containing one option:
%    options.stepsize (positive double; default: 1e-4).
%
% If the gradient cannot be computed on 'problem', a warning is issued.
%
% Output:
% 
% Returns a function handle, encapsulating a generic finite difference
% approximation of the Hessian of the problem cost. The finite difference
% is based on computations of the gradient.
% 
% The returned hessfun has this calling pattern:
% 
%   function hessfd = hessfun(x, xdot)
%   function hessfd = hessfun(x, xdot, storedb)
%   function hessfd = hessfun(x, xdot, storedb, key)
% 
% x is a point on the manifold problem.M, xdot is a tangent vector to that
% manifold at x, storedb is a StoreDB object, and key is the StoreDB key to
% point x.
%
% Usage:
%
% Typically, the user will set problem.M and other fields to define the
% cost and the gradient (typically, problem.cost and problem.grad or
% problem.egrad). Then, to use this generic purpose Hessian approximation:
%
%   problem.approxhess = approxhessianFD(problem, options);
%
% See also: trustregions

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, April 8, 2015.
% Contributors: 
% Change log: 
%
%   Feb. 19, 2015 (NB):
%       It is sufficient to ensure positive radial linearity to guarantee
%       (together with other assumptions) that this approximation of the
%       Hessian will confer global convergence to the trust-regions method.
%       Formerly, in-code comments referred to the necessity of having
%       complete radial linearity, and that this was harder to achieve.
%       This appears not to be necessary after all, which simplifies the
%       code.
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%   April 8, 2015 (NB):
%       Changed to approxhessianFD, which now returns a function handle
%       that encapsulates the getHessianFD functionality. Will be better
%       aligned with the other Hessian approximations to come (which may
%       want to use storedb.internal), and now allows specifying the step
%       size.

    % This Hessian approximation is based on the gradient:
    % check availability.
    if ~canGetGradient(problem)
        warning('manopt:approxinversehessianBFGS:nogradient', ...
                'approxinversehessianBFGS requires the gradient to be computable.');
    end

    %     % Set local defaults here, and merge with user options, if any.
    %     localdefaults.stepsize = 1e-4;
    %     if ~exist('options', 'var') || isempty(options)
    %         options = struct();
    %     end
    %     options = mergeOptions(localdefaults, options);
    
    %     % Finite-difference parameter: how far do we look?
    %     stepsize = options.stepsize;
                   
    % Build and return the function handle here. This extra construct via
    % funhandle makes it possible to make storedb and key optional.
    hessfun = @funhandle;
    function [inversehess storedb] = funhandle(x, xdot, storedb, key)
        % Allow omission of the key, and even of storedb.
        if ~exist('key', 'var')
            if ~exist('storedb', 'var')
                storedb = StoreDB();
            end
            key = storedb.getNewKey();
        end 
        [inversehess storedb] = inversehessianBFGS(problem, x, xdot, options, storedb, key);
    end
    
end


function [dsd storedb] = inversehessianBFGS(problem, x, xdot, options, storedb, key)
% This function does the actual work.
%
% Original code: Dec. 30, 2012 (NB).
	
	% Extract the input vector norm.
    norm_xdot = problem.M.norm(x, xdot);
    
    % First, check whether the step xdot is not too small.
    if norm_xdot < eps
        dsd = problem.M.zerovec(x);
        return;
    end
    
    % Check whether this is the first call.
    if ~isfield(storedb.internal, 'cg') %
        dsd = xdot;
        
        nsmax = 100; % options.nsmax;  % BM: should be supplied for the limited memory version
        storedb.internal.BFGS.ns = 0;
        storedb.internal.BFGS.alpha = zeros(nsmax, 1);
        storedb.internal.BFGS.beta = zeros(nsmax, 1);
        storedb.internal.BFGS.sstore = cell(nsmax);
        storedb.internal.BFGS.alphatmp = 0;
        storedb.internal.BFGS.b = 0;
        storedb.internal.BFGS.b0 = 0;
        
        return;
    end
    
    desc_dir = storedb.internal.cg.stepdirection; % search dirction at origin
    xc = storedb.internal.cg.steporigin;% x orgin;
    xt = storedb.internal.cg.steptarget;% newx;
    lambda = storedb.internal.cg.stepsize; 
    keyc = storedb.internal.cg.steporiginkey;
    keyt = storedb.internal.cg.steptargetkey;
    
    % Computing s, y, and go
    s = problem.M.transp(xc, xt, desc_dir);  % Transport s to newx
    gc = getGradient(problem, xc, storedb, keyc); % Gradient origin
    gc = problem.M.transp(xc, xt, gc); % Transport gradient from origin to newx
    gt = getGradient(problem, xt, storedb, keyt); % Get new gradient
    y = problem.M.lincomb(xt, 1, gt, -1, gc);  % Gradient difference
    yts = problem.M.inner(xt, s, y); % Inner product between y and s
    go = problem.M.inner(xt, s, gc); % Inner product between y and s
    
    
    
    % Options
    ns = storedb.internal.BFGS.ns;
    nsmax = 100; % options.nsmax; % BM: should be supplied for the limited memory version
    
   
    %   Restart if y'*s is not positive or we're out of room
    if (yts <= 0) || (ns == nsmax)
        disp(' loss of positivity or storage');
        ns = 0;
        storedb.internal.BFGS.ns = 0;
        storedb.internal.BFGS.alpha = zeros(nsmax, 1);
        storedb.internal.BFGS.beta = zeros(nsmax, 1);
        storedb.internal.BFGS.sstore = cell(nsmax);
    else
        ns = ns + 1;
        
        % Store
        storedb.internal.BFGS.ns = ns;
        storedb.internal.BFGS.sstore(ns) = s; % BM: Check this?
        
        if(ns > 1)
            alpha = storedb.internal.BFGS.alpha;
            beta = storedb.internal.BFGS.beta;
            b0 = storedb.internal.BFGS.b0;
            b = storedb.internal.BFGS.b;
            alphatmp = storedb.internal.BFGS.alphatmp;
            
            alpha(ns - 1)= alphatmp;
            beta(ns - 1) = b0 / (b*lambda);
            
            % Store
            storedb.internal.BFGS.alpha = alpha;
            storedb.internal.BFGS.beta = beta;
    
        end
    end
    
    
    % Compute the Hessian application
    dsdp =  problem.M.lincomb(xt, -1, gt); % -gt;
    if (ns > 1)
        dsdp = bfgsw(problem, xt, xdot, storedb);
    end
    
    xi = dsdp;
    b0 = -1/yts;
    
    % Verify these quantities
    
    %     zeta = problem.M.lincomb(xt, (1 - 1/lambda), s,  1, xi);  % (1 - 1/lambda)*s + xi;
    %     a1 = b0*b0*(problem.M.inner(xt, zeta, y)); % b0*b0*(zeta'*y);
    
    a1 = -b0*(1 - 1/lambda) + b0*b0*(problem.M.inner(xt, xi, y)); % -b0*(1 - 1/lambda)+b0*b0*y'*xi;
    a = - problem.M.inner(xt, gc, problem.M.lincomb(xt, a1, s, b0, xi) ); % -(a1*s + b0*xi)'*gc;
    alphatmp = a1 + 2*a/go;
    b = -b0*go;
    
    % Compute the search direction
    dsd =  problem.M.lincomb(xt, a, s, b, xi); % dsd = a*s + b*xi;
    
    
    % Store
    storedb.internal.BFGS.alphatmp = alphatmp;
    storedb.internal.BFGS.b = b;
    storedb.internal.BFGS.b0 = b0;
    % storedb.internal.BFGS.ns = ns; % already updated
    % storedb.internal.BFGS.alpha = alpha;  % already updated
    % storedb.internal.BFGS.beta = beta; % already updated
    % storedb.internal.BFGS.sstore = []; % already updated
         
end

function dnewt = bfgsw(problem, x, xdot, storedb)
    ns = storedb.internal.BFGS.ns;
    sstore = storedb.internal.BFGS.sstore; % cell of cells
    alpha = storedb.internal.BFGS.alpha;
    beta = storedb.internal.BFGS.beta;
    
    dnewt = xdot;
    if ns <= 1 
        return; 
    end;
    dnewt = xdot; 
    
    
    % BM: Could this be accelerated?
    fullsigma = zeros(ns, 0);
    projsstore = cell(ns);
    for ii = 1: ns 
        sii = sstore(ii);
        sii = problem.M.proj(x, sii); % BM: proj or should we use problem.M.transp?
        projsstore(ii) = sii;
        fullsigma(ii) = problem.M.inner(x, sii, dsd);
    end
    
    
    sigma = fullsigma(1:ns-1); %    sigma = sstore(:,1:ns-1)'*dsd; 
    gamma1 = alpha(1:ns-1).*sigma;
    gamma2 = beta(1:ns-1).*sigma;
    gamma3 = gamma1 + beta(1:ns-1).*(fullsigma(2:ns)); %gamma1 + beta(1:ns-1).*(sstore(:,2:ns)'*dsd);
    delta = gamma2(1:ns-2) + gamma3(2:ns-1);
    
    %
    %     dnewt = dnewt + gamma3(1)*sstore(:,1) + gamma2(ns-1)*sstore(:,ns);
    dnewt = problem.M.lincomb(x, 1, dnewt, 1,...
                              problem.M.lincomb(x, gamma3(1), sstore(1), gamma2(ns-1), sstore(ns))...
                              );
    if(ns <= 2) 
        return; 
    end
    
    
    % BM: Could this be accelerated?
    %     dnewt = dnewt + sstore(1:n,2:ns-1)*delta(1:ns-2);
    for jj = 2: ns - 1
        dnewt = problem.M.lincomb(x, 1, dnewt, delta(jj -1),...
            projsstore(jj));
    end
end

