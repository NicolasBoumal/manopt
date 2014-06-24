function basicexample
 
    import manopt.solvers.trustregions.*;
    import manopt.manifolds.sphere.*;
    import manopt.tools.*;
    
    % This if-block is there to verify that Manopt was indeed added to the
    % Matlab path /before/ the present script was loaded in memory. This is
    % important so that the 'import' commands above work properly.
    if isempty(which('spherefactory'))
        presdir = pwd;
        cd ..;
        manoptdir = pwd;
        cd(presdir);
        
        cmd = sprintf('addpath(''%s''), clear %s;', manoptdir, mfilename);
        warning(['Prior to executing this function, the manopt parent ' ...
                 'directory must be added to the path. To do this, we ' ...
                 'now execute in the command prompt: %s'], cmd);%#ok<SPWRN>
        evalin('base', cmd);
        error('You may now re-execute the function %s.', mfilename);
    end
    
    % Generate the problem data.
    n = 1000;
    A = randn(n);
    A = .5*(A+A');
    
    % Create the problem structure.
    manifold = spherefactory(n);
    problem.M = manifold;
    
    % Define the problem cost function and its gradient.
    problem.cost = @(x) -x'*(A*x);
    problem.grad = @(x) manifold.proj(x, -2*A*x);
    
    % Numerically check gradient consistency.
    checkgradient(problem);
 
    % Solve.
    % The trust-regions algorithm requires the Hessian. Since we do not
    % provide it, it will go for a standard approximation of it. The first
    % instruction tells Manopt not to issue a warning when this happens.
    warning('off', 'manopt:getHessian:approx');
    [x xcost info] = trustregions(problem); %#ok<ASGLU>
    
    % Display some statistics.
    figure;
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration #');
    ylabel('Gradient norm');
    title('Convergence of the trust-regions algorithm on the sphere');
    
end
