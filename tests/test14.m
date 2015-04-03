function test14()
% Test for complexcircle geometry (Z2 synchronization)
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    n = 1000;
    z = 2*randi(2, n, 1)-3;
    
    % Exact data matrix
    H = z*z' - eye(n);
    
    % Add noise: switch a fraction p of the measurements
    p = .485;
    mask = rand(n) <= p;
    mask = triu(mask, 1);
    mask = mask + mask';
    mask = logical(mask);
    H(mask) = -H(mask);
    
    problem.M = complexcirclefactory(n);
    
    problem.cost = @(z) -.5*real(z'*H*z);
    problem.grad = @(z) problem.M.proj(z, -H*z);
    
    checkgradient(problem); pause;

    % Compute eigenvector initial guess
    [v, d] = eigs(H, 1); %#ok<NASGU>
    z0 = sign(v);
    
    % TODO investigate this: z0 seems to be a "point selle"
    z0 = problem.M.retr(z0, problem.M.randvec(z0), .01*n);
    
    options.Delta_bar = 10*pi*n;
    [zopt, costopt, info] = trustregions(problem, z0, options); %#ok<NASGU,ASGLU>
%     [zopt costopt] = steepestdescent(problem, z0);
    
    % realign zopt (since this is all up to a phase shift)
    [~, ~, V] = svd([real(zopt) imag(zopt)], 'econ');
    rho = V(1, 1) - 1i*V(2, 1);
    zopt = rho*zopt;
    
    % Plot it
    t = linspace(0, 2*pi, 101);
    plot(cos(t), sin(t), 'r-', real(zopt), imag(zopt), 'b.'); axis equal;
    xlim([-1.1, 1.1]);
    ylim([-1.1, 1.1]);
    pbaspect([1 1 1]);
    axis off;
    
    % Now decide a class (1 or -1) for each zopt(i)
    zopt = sign(real(zopt));
    
    abs(z'*zopt)
    
    % Compare with the eigenvector method
    zeig = sign(v);
    abs(zeig'*z)
    
%     keyboard;

end