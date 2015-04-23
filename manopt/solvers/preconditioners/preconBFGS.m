function preconfun =  preconBFGS(problem, options)
% Approximate inverse Hessian based on a BFGS update rule.
%
% function preconfun = preconBFGS(problem)
% function preconfun = preconBFGS(problem, options)
%
% Input:
%
% A Manopt problem structure (already containing the manifold and enough
% information to compute the cost gradient) and an options structure
% (optional). It future versions, it will be possible to use options to
% limit memory usage, for example.
%
% If the gradient cannot be computed on 'problem', a warning is issued.
%
% Output:
% 
% Returns a function handle, encapsulating a generic preconditioner (that
% is, an approximation of the inverse of the Hessian of the problem cost.)
% The approximation is based on a generalization of the BFGS method.
% 
% The returned preconfun has this calling pattern:
% 
%   function Pxdot = preconfun(x, xdot)
%   function Pxdot = preconfun(x, xdot, storedb)
%   function Pxdot = preconfun(x, xdot, storedb, key)
% 
% x is a point on the manifold problem.M, xdot is a tangent vector to that
% manifold at x, storedb is a StoreDB object, and key is the StoreDB key to
% point x.
%
% Usage:
%
% Typically, the user will set problem.M and other fields to define the
% cost and the gradient (typically, problem.cost and problem.grad or
% problem.egrad). Then, to use this generic purpose preconditioner:
%
%   problem.precon = preconBFGS(problem, options);
%
% See also: conjugategradient

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, April 22, 2015.
% Contributors: Nicolas Boumal
% Change log: 

% TODO : For now, only works with CG solver, for two reasons:
%        1) looks up storedb.internal.cg : need to be more general.
%        2) assumes exactly one call to precon per iteration, so that the
%           inverse Hessian approximation is updated at each call.
% TODO : comment on the non-usability of hessianspectrum because of storedb
%        and think of a workaround.

    % This preconditioner requires the gradient: check availability.
    if ~canGetGradient(problem)
        warning('manopt:preconBFGS:nogradient', ...
                'preconBFGS requires the gradient to be computable.');
    end
    
    if ~exist('options', 'var')
        options = struct();
    end
                   
    % Build and return the function handle here. This extra construct via
    % funhandle makes it possible to make storedb and key optional.
    % Note that for this preconditioner, it would make little sense to call
    % it without a storedb, since it is based on a history of the
    % iterations.
    preconfun = @funhandle;
    function Pxdot = funhandle(x, xdot, storedb, key)
        % Allow omission of the key, and even of storedb.
        if ~exist('key', 'var')
            if ~exist('storedb', 'var')
                storedb = StoreDB();
            end
            key = storedb.getNewKey();
        end 
        Pxdot = BFGShelper(problem, x, xdot, options, storedb, key);
    end
    
end


function Pxdot = BFGShelper(problem, x, xdot, options, storedb, key) %#ok<INUSD,INUSL>
% This function does the actual work.
	
	% Extract the input vector norm.
    norm_xdot = problem.M.norm(x, xdot);
    
    % First, check whether the step xdot is not too small.
    if norm_xdot < eps
        Pxdot = problem.M.zerovec(x);
        return;
    end
    
    % If this is the first call, use the identity as approximation.
    if ~isfield(storedb.internal, 'cg')
        Pxdot = xdot;
        
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
    if (yts <= 0) || (ns >= nsmax)
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
        storedb.internal.BFGS.sstore{ns} = s; % BM: Check this?
        
        if(ns > 1)
            alpha = storedb.internal.BFGS.alpha;
            beta = storedb.internal.BFGS.beta;
            b0 = storedb.internal.BFGS.b0;
            b = storedb.internal.BFGS.b;
            alphatmp = storedb.internal.BFGS.alphatmp;
            
            alpha(ns - 1) = alphatmp;
            beta(ns - 1)  = b0 / (b*lambda);
            
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
    Pxdot =  problem.M.lincomb(xt, a, s, b, xi); % dsd = a*s + b*xi;
    
    
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
    for ii = 1 : ns 
        sii = sstore{ii};
        sii = problem.M.proj(x, sii); % BM: proj or should we use problem.M.transp? NB : Is it a tangent vector, or could it not be one? If tangent, what is the root point of the vector? And what do you want the root to be?
        projsstore{ii} = sii;
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
                              problem.M.lincomb(x, gamma3(1), sstore{1}, gamma2(ns-1), sstore{ns})...
                              );
    if(ns <= 2) 
        return; 
    end
    
    
    % BM: Could this be accelerated?
    %     dnewt = dnewt + sstore(1:n,2:ns-1)*delta(1:ns-2);
    for jj = 2: ns - 1
        dnewt = problem.M.lincomb(x, 1, dnewt, delta(jj -1),...
            projsstore{jj});
    end
end

