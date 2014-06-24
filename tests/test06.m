function R = test06()
% function R = test06()
%
% Test case for the Stiefel manifold, applied to Cryo-EM problem.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    % Load the data
    data = load('cryoemdata500_SNR03.mat', 'S');
    S = data.S;
    N = size(S, 1)/2;
    
    % Pick the manifold
    problem.M = stiefelfactory(3, 2, N);
    warning('off', 'manopt:stiefel:exp');
    
    % Build an initial guess based on dominant eigenspaces of S
    [V, D] = eigs(S, 3);
    V = V*sqrt(D);
    Z0 = problem.M.retr(reshape(V', [3 2 N]), problem.M.zerovec());

    % Define the problem cost function
    problem.cost = @cost;
    function [val store] = cost(Z, store)
        Z = reshape(Z, [3 2*N]);
        
        if ~isfield(store, 'ZS')
            store.ZS = Z*S;
        end
        ZS = store.ZS;
           
        ZZt = Z*Z';
        term1 = norm(ZZt, 'fro')^2;
        term2 = ZS(:)'*Z(:);
        val = .25*term1-.5*term2;
    end

    % And its gradient
    problem.grad = @grad;
    function [G store] = grad(Z, store)
        Z = reshape(Z, [3 2*N]);
        
        if ~isfield(store, 'ZS')
            store.ZS = Z*S;
        end
        ZS = store.ZS;
        
        ZZt = Z*Z';
        
        G = ZZt*Z-ZS;
        
        Z = reshape(Z, [3 2 N]);
        G = reshape(G, [3 2 N]);
        G = problem.M.proj(Z, G);
    end
    
    % Check differentials consistency.
%     checkgradient(problem);

    % Solve with trust-regions and FD approximation of the Hessian
    warning('off', 'manopt:getHessian:approx');
    Z = trustregions(problem, Z0);
    
    % Post-processing: turn the 2-frames in R^3 into rotation matrices by
    % computing the third vector (vector cross-product).
    R = zeros(3, 3, N);
    for i = 1 : N
        R(:, 1:2, i) = Z(:, :, i);
        R(:,  3 , i) = cross(R(:, 1, i), R(:, 2, i));
    end
    
    save('R.mat', 'R');
    
    Z0 = reshape(Z0, [3 2*N]);
    Z  = reshape(Z,  [3 2*N]);
    fprintf('||S-Z0''Z0|| = %g\n', norm(S-Z0'*Z0, 'fro'));
    fprintf('||S-Z''Z|| = %g\n', norm(S-Z'*Z, 'fro'));
    
end
