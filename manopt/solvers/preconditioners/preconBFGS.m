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


    % We change the sign temporarily. We change it back too. 
    xdot = problem.M.lincomb(x, -1, xdot);
    
    
    % Extract the input vector norm.
    norm_xdot = problem.M.norm(x, xdot);
    
    % First, check whether the step xdot is not too small.
    if norm_xdot < eps
        Pxdot = problem.M.zerovec(x);
        return;
    end
    
    % If this is the first call, use the identity as the Hessian 
    % approximation.
    if ~isfield(storedb.internal, 'cg')
        fprintf('Identity operation\n');
        Pxdot = xdot; % Identity operation
        
        nsmax = 500; % options.nsmax;  % BM: should be supplied for the limited memory version
        storedb.internal.BFGS.ns = 0;
        storedb.internal.BFGS.alpha = zeros(nsmax, 1);
        storedb.internal.BFGS.beta = zeros(nsmax, 1);
        storedb.internal.BFGS.sstore = cell(nsmax);
        storedb.internal.BFGS.xstore = cell(nsmax);
        storedb.internal.BFGS.alphatmp = 0;
        storedb.internal.BFGS.b = 0;
        storedb.internal.BFGS.b0 = 0;
        
        Pxdot = problem.M.lincomb(x, -1, Pxdot);
        return;
    end
    
    desc_dir = storedb.internal.cg.stepdirection; % search dirction at origin
    xc = storedb.internal.cg.steporigin;% x orgin;
    xt = storedb.internal.cg.steptarget;% newx;
    lambda = storedb.internal.cg.stepsize/problem.M.norm(xc,desc_dir); % BM: should be 1 or less? verify  % storedb.internal.cg.stepsize; 
    keyc = storedb.internal.cg.steporiginkey;
    keyt = storedb.internal.cg.steptargetkey;

    
    % Computing s, y, yts and go
    % At xc
    step = problem.M.lincomb(xc, lambda, desc_dir);
    gc = getGradient(problem, xc, storedb, keyc); % Gradient origin
    
    % At xt
    s = problem.M.transp(xc, xt, step);  % Transport s to newx. BM: corrrect. 
    gc = problem.M.transp(xc, xt, gc); % Transport gradient from origin to newx
    gt = getGradient(problem, xt, storedb, keyt); % Get new gradient
    y = problem.M.lincomb(xt, 1, gt, -1, gc);  % Gradient difference
    yts = problem.M.inner(xt, s, y); % Inner product between y and s
    go = problem.M.inner(xt, s, gc); % Inner product between gc and s
    
    
    
    % Options
    ns = storedb.internal.BFGS.ns;
    nsmax = 500; % options.nsmax; % BM: should be supplied for the limited memory version
    
   
    %   Restart if y'*s is not positive or we're out of room
    if (yts <= 0) || (ns >= nsmax)
        disp('loss of positivity or storage');
        ns = 0;
        storedb.internal.BFGS.ns = 0;
        storedb.internal.BFGS.alpha = zeros(nsmax, 1);
        storedb.internal.BFGS.beta = zeros(nsmax, 1);
        storedb.internal.BFGS.sstore = cell(nsmax);
        storedb.internal.BFGS.xstore = cell(nsmax);
    else
        ns = ns + 1;
        
        % Store
        storedb.internal.BFGS.ns = ns;
        storedb.internal.BFGS.sstore{ns} = s; 
        storedb.internal.BFGS.xstore{ns} = xc;
        
        
        if(ns > 1)
            alpha = storedb.internal.BFGS.alpha;
            beta = storedb.internal.BFGS.beta;
            b0 = storedb.internal.BFGS.b0;
            b = storedb.internal.BFGS.b;
            alphatmp = storedb.internal.BFGS.alphatmp;
            
            alpha(ns - 1) = alphatmp;
            beta(ns - 1)  = b0/(b*lambda); %BM: correct upto sign % CTK: b0/(b*lambda)
            
            % Store
            storedb.internal.BFGS.alpha = alpha;
            storedb.internal.BFGS.beta = beta;
        end
    end
    
  
    % Compute the Hessian application
    % Here gc of Kelly's code transforms to gt for us
    Pxdot = xdot;% xdot; -gt
    dsdp =  problem.M.lincomb(xt, -1, gt); % CTK:-gc
    if (ns > 1)
        dsdp = bfgsw(problem, xt, Pxdot, storedb);
    end
    
    if ns == 0
        fprintf('Identity operation\n');
        Pxdot = xdot;% xdot; % Identity operation
        pause;
    else
        % Here gc of CTK's code transforms to gt for us.
        
        xi = problem.M.lincomb(xt, -1, dsdp); % xi = -dsdp;
        b0 = -1/yts; % Should be a negative quantity

        % BM: verify these quantities.
        % CTK: -b0*(1 - 1/lambda)+b0*b0*y'*xi. Below is the direct
        % implementation.
        % a1 = -b0*(1 - 1/lambda) + b0*b0*(problem.M.inner(xt, xi, y));
        % However, lambda for us does not go to 1 (why ?) and 
        % we modify the update.
        a1 =  b0*b0*(problem.M.inner(xt, xi, y)); 
        
        % CTK: -(a1*s + b0*xi)'*gc;
        a = -problem.M.inner(xt, gt, problem.M.lincomb(xt, a1, s, b0, xi));
        alphatmp = a1 + 2*a/go;
        b = -b0*go; % Bc
                
        % Compute search direction
        Pxdot =  problem.M.lincomb(xt, a, s, b, xi); % CTK: dsd = a*s + b*xi;
        
         
        % Store
        storedb.internal.BFGS.alphatmp = alphatmp;
        storedb.internal.BFGS.b = b;
        storedb.internal.BFGS.b0 = b0;
        % storedb.internal.BFGS.ns = ns; % already updated
        % storedb.internal.BFGS.alpha = alpha;  % already updated
        % storedb.internal.BFGS.beta = beta; % already updated
        % storedb.internal.BFGS.sstore = []; % already updated
    end
    
    % Check whether we have a positive definite preconditioner
    if problem.M.inner(xt, Pxdot, xdot) <= 1.d-6*(problem.M.norm(xt, Pxdot))*(problem.M.norm(xt, xdot))
        % The Hessian storage is suspect, reset it
        Pxdot =  problem.M.lincomb(xt, 1,  xdot); %xdot or -gt % Identity operation
        fprintf('loss of descent\n');
        storedb.internal.BFGS.ns = 0;
        storedb.internal.BFGS.alpha = zeros(nsmax, 1);
        storedb.internal.BFGS.beta = zeros(nsmax, 1);
        storedb.internal.BFGS.sstore = cell(nsmax);
        storedb.internal.BFGS.xstore = cell(nsmax);
    end
    
    
    % Change the sign back.
    Pxdot = problem.M.lincomb(x, -1, Pxdot);
end

function dnewt = bfgsw(problem, x, dsd, storedb)
    ns = storedb.internal.BFGS.ns;
    sstore = storedb.internal.BFGS.sstore; 
    xstore = storedb.internal.BFGS.xstore; 
    alpha = storedb.internal.BFGS.alpha;
    beta = storedb.internal.BFGS.beta;
    
    dnewt = dsd;
    if ns <= 1 
        return; 
    end;
    dnewt = dsd; 
    
    
    % BM: Could this be accelerated?
    fullsigma = zeros(ns, 1);
    transpsstore = cell(ns);
    for ii = 1 : ns 
        sii = sstore{ii};
        xii = xstore{ii};
        sii = problem.M.transp(xii, x, sii);
        transpsstore{ii} = sii;
        fullsigma(ii) = problem.M.inner(x, sii, dsd);
    end
    
    sigma = fullsigma(1:ns-1); % CTK: sigma = sstore(:,1:ns-1)'*dsd; 
    gamma1 = alpha(1:ns-1).*sigma;
    gamma2 = beta(1:ns-1).*sigma;
    gamma3 = gamma1 + beta(1:ns-1).*(fullsigma(2:ns)); % CTK: gamma1 + beta(1:ns-1).*(sstore(:,2:ns)'*dsd);
    delta = gamma2(1:ns-2) + gamma3(2:ns-1);
    
    % CTK: dnewt = dnewt + gamma3(1)*sstore(:,1) + gamma2(ns-1)*sstore(:,ns);
    dnewt = problem.M.lincomb(x, 1, dnewt, 1,...
                              problem.M.lincomb(x, gamma3(1), sstore{1}, gamma2(ns-1), sstore{ns}));
    if(ns <= 2) 
        return; 
    end
    
    % BM: Could this be accelerated?
    % CTK: dnewt = dnewt + sstore(1:n,2:ns-1)*delta(1:ns-2);
    for jj = 2: ns - 1
        dnewt = problem.M.lincomb(x, 1, dnewt, delta(jj -1),...
            transpsstore{jj});
    end
end

