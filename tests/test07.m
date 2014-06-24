function test07()
% function test07()
%
% Computes a robust mean of angles
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
    
    % Generate the data
    N = 250;
    theta_true = pi*(2*rand(1)-1);
    thetas = zeros(N,1);
    p_true = .25;
    for i = 1 : N
        if rand(1) < p_true
            thetas(i) = theta_true + .3*randn(1);
        else
            thetas(i) = pi*(2*rand(1)-1);
        end
    end
    X = [cos(thetas') ; sin(thetas)'];
    
    % Pick the manifold
    problem.M = spherefactory(2);

    % Parameters
    p = p_true;
    kappa = 3;
    c2 = besseli(0, 2*kappa);
    
    % Define the problem cost function
    problem.cost = @cost;
    function [val store] = cost(x, store)
        
        if ~isfield(store, 'fi')
            store.fi = (p/c2)*exp(2*kappa*X'*x);
        end
        fi = store.fi;
           
        val = -sum(log(fi + (1-p)));
        
    end

    % And its gradient
    problem.grad = @grad;
    function [g store] = grad(x, store)
        
        if ~isfield(store, 'fi')
            store.fi = (p/c2)*exp(2*kappa*X'*x);
        end
        fi = store.fi;
        
        g = -X*(2*kappa*(fi./(fi+(1-p))));
        g = g - (x'*g)*x;
        
    end
    
    % Check differentials consistency.
    % checkgradient(problem);

    % Solve with trust-regions and FD approximation of the Hessian
    warning('off', 'manopt:getHessian:approx');
    
    % Test many random initial guess
%     best_cost = inf;
%     best_x = [];
%     for i = 1 : 5
%         [x cst] = trustregions(problem);
%         if cst < best_cost
%             best_cost = cst;
%             best_x = x;
%         end
%     end
%     theta_found = angle(best_x(1)+1i*best_x(2));

    % Do a histogram of the data and select the initial guess as the peak
    [freqs, bins] = hist(thetas, 30);
    [~, ind] = max(freqs);
    x0 = [cos(bins(ind)) ; sin(bins(ind))];
    x = trustregions(problem, x0);
    theta_found = angle(x(1)+1i*x(2));
    
    fprintf('True theta: %g\nHist theta: %g\nFound theta: %g\n', theta_true, angle(x0(1)+1i*x0(2)), theta_found);
    
    subplot(2,1,1);
    hist(thetas, 30);
    xlim([-pi pi]);
    
    th = linspace(-pi, pi);
    fth = zeros(size(th));
    for i = 1 : length(th)
        fth(i) = cost([cos(th(i));sin(th(i))], struct());
    end
    subplot(2,1,2);
    plot(th, -fth);
    xlim([-pi pi]);
    
%     keyboard;
    
end
