function [value] = test21(epsilon, A, B, C, D)
% test for complex sphere
% See notes October 11, 2012.
%
% fzero(@(x) test21(x), [.1 1]) searches the value of epsilon such that we
% barely destabilize the system with a norm epsilon perturbation.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    clc;
    reset(RandStream.getDefaultStream);
    randnfoo = randn(1234567, 1); %#ok<NASGU>
    
    % define data
    if ~exist('A', 'var') || ~exist('B', 'var') || ~exist('C', 'var') || ~exist('D', 'var')
        if 1
            data = load('code_hinfpaper_examples\only_mats\closed-loop-order-3-HE6.mat');
            A = data.A;
            n = size(A, 1); assert(size(A, 2) == n);
            B = data.B;
            p = size(B, 2); assert(size(B, 1) == n);
            C = data.C;
            m = size(C, 1); assert(size(C, 2) == n);
            if isfield(data, 'D')
                D = data.D;
                assert(all(size(D) == [m p]));
            else
                D = zeros(m, p);
            end
        else
            n = 50;
            m = 5;
            p = 7;
            A = randn(n) - 2*sqrt(n)*eye(n);
            B = randn(n, p);
            C = randn(m, n);
    %         D = randn(m, p);
            D = zeros(m, p);
        end
    else
        n = size(A, 1); assert(size(A, 2) == n);
        p = size(B, 2); assert(size(B, 1) == n);
        m = size(C, 1); assert(size(C, 2) == n);
        assert(all(size(D) == [m p]));
    end
    
    if ~exist('epsilon', 'var')
        epsilon = 1;
    end
    
    
    
    % set to true if want to use sparse version of eig: eigs
    useeigs = false;
    
    % Create the problem structure
    M = spherecomplexfactory(p, m);
    problem.M = M;
    
    % Define the problem cost function
    problem.cost = @cost;
    function f = cost(X)
        K = inv(eye(m)/epsilon - D*X);
        gX = A + B*X*K*C;
        
        if useeigs
            [~, lam, ~] = eigs(gX , 1, 'LR');
            f = -real(lam);
        else
            spec = eig(gX);
            f = -max(real(spec));
        end
        
    end
    
    problem.grad = @grad;
    function G = grad(X)
        K = inv(eye(m)/epsilon - D*X);
        gX = A + B*X*K*C;
        
        if useeigs
            [y, lamy, ~] = eigs(gX , 1, 'LR');
            [x, lamx, ~] = eigs(gX', 1, 'LR');
            assert(abs(conj(lamx)-lamy) < 1e-10);
            % Alternative to get the left eigenvector (don't know which is
            % better)
            % [x, ~,   ~] = eigs(gX', 1, conj(lam));
        else
            [V DD] = eig(gX);
            [f I] = max(real(diag(DD)));
            lam = DD(I, I);
            y = V(:, I);
            [V DD] = eig(gX');
            [d, I] = min(abs(diag(DD)-conj(lam)));
            assert(d < 1e-10);
            x = V(:, I);
        end
        
        Z = (x*y')/(y'*x);
        G = (eye(p) + D'*K'*X')*B'*Z*C'*K';
%         G = (  ((eye(m)/epsilon - D*X)\C)*Z'*B*(eye(p) + X*((eye(m)/epsilon - D*X)\D))  )';
        G = M.proj(X, -G);
    end
    
    % If the optimization algorithms require Hessians, since we do not
    % provide it, it will go for a standard approximation of it. This line
    % tells Matlab not to issue a warning when this happens.
    warning('off', 'manopt:getHessian:approx');
    
    % Check gradient consistency.
%     checkgradient(problem);

    options.statsfun = @statsfun;

    % Solve

%     [X bestcost info] = steepestdescent(problem, [], options);
    [X bestcost info] = trustregions(problem, [], options);
%     [X bestcost info] = neldermead(problem);

% keyboard;

    value = -bestcost;
    
end


function stats = statsfun(problem, X, stats)

    return;

    persistent xlims ylims;
    if isempty(xlims) || isempty(ylims)
        spectrum = eig(full(A));
        xlims = [min(real(spectrum)) max(real(spectrum))];
        xlims = mean(xlims) + diff(xlims) * [-1 1];
        ylims = [min(imag(spectrum)) max(imag(spectrum))];
        ylims = mean(ylims) + diff(ylims) * [-1 1];
    end

    figure(1);
    hold on;
    K = inv(eye(m)/epsilon - D*X);
    gX = A + B*X*K*C;
    spectrum = eig(gX);
    plot(real(spectrum), imag(spectrum), '.');
    hold off;
    xlim(xlims);
    ylim(ylims);
    drawnow;
end