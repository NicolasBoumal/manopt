function lowrank_dist_completion( )
% lowrank_dist_completion()
% Low-rank Euclidean distance matrix completion
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, April 06, 2015.
% Contributors:
% Change log:
    clear all; clc; close all;
    
    n = 500; % Number of points
    r = 5; % Embedding dimension
    
    % Fraction of unknown distances
    fractionOfUnknown = 0.8;
    
    
    Yo = randn(n,r); % True embedding
    trueDists = pdist(Yo)'.^2; % True distances
    
    % Comment out this line if you don't want to add noise
    trueDists = trueDists + 0.01 * std(trueDists) * randn(size(trueDists)); % add noise
    
    % Compute all pair of indices
    H = tril(true(n),-1);
    [I, J] = ind2sub([n,n],find(H(:)));
    clear 'H';
    
    % Train data
    train = false(length(trueDists),1);
    train(1:floor(length(trueDists)*(1- fractionOfUnknown))) = true;
    train = train(randperm(length(train)));
    
    data_train.rows = I(train);
    data_train.cols = J(train);
    data_train.entries = trueDists(train);
    data_train.nentries = length(data_train.entries);
    
    % Test data
    data_test.nentries = 1*data_train.nentries; % Depends how big data can we handle.
    test = false(length(trueDists),1);
    test(1 : floor(data_test.nentries)) = true;
    test = test(randperm(length(test)));
    data_test.rows = I(test);
    data_test.cols = J(test);
    data_test.entries = trueDists(test);
    
   
    %% Rank search algorithm
    r_initial = 1;
    rmax = n;
    N = data_train.nentries; % Number of known distances
    
    EIJ = speye(n);
    EIJ = EIJ(:, data_train.rows) - EIJ(:, data_train.cols);
    
    rr = r_initial;
    Y = randn(n, rr);
    
    time = [];
    cost = [];
    test_error = [];
    % Optimization main loop
    while (rr <= rmax), % When r = n a global min is attained for sure.
        fprintf('>> Rank %d <<\n', rr);
        if (rr > r_initial),
            if isempty(restartDir), % If no restart dir avail. do random restart
                
                disp('No restart dir available, random restart is performed');
                Y = randn(n, rr);
                
            else % Perform line-search based on the restart direction
                
                disp('>> Line-search with restart direction');
                Y(:, rr) = 0; % Append a column of zeroes
                
                Z = Y(data_train.rows, :) - Y(data_train.cols,:);
                estimDists = sum(Z.^2, 2);
                errors = (estimDists - data_train.entries);
                
                grad_Y = EIJ * sparse(1:N,1:N,2 * errors / N,N,N) * Z;
                
                costBefore = mean(errors.^2);
                fprintf('>> Cost before = %f\n',costBefore);
                
                step = N/(n^2); % A very rough estimate of the Lipscitz constant
                for i = 1 : 20, % 20 backtracking to find a descent direction.
                    % Update
                    Y(:,rr) = step*restartDir;
                    
                    % Compute cost
                    Z = Y(data_train.rows, :) - Y(data_train.cols,:);
                    estimDists = sum(Z.^2,2);
                    errors = (estimDists - data_train.entries);
                    
                    costAfter = mean(errors.^2);
                    fprintf('>> Cost after = %f\n',costAfter);
                    
                    % Armijo condition
                    armijo = (costAfter - costBefore) <= 0.5 * step * (restartDir'*grad_Y(:,rr));
                    if armijo,
                        break;
                    else
                        step = step/2;
                    end
                    
                end
                
                % Check for sufficient decrease
                if (costAfter >= costBefore) || abs(costAfter - costBefore) < 1e-8,
                    disp('Decrease is not sufficient, random restart');
                    Y = randn(n, rr);
                end
                
            end
            
        end
        
        % Run fixed-rank optimization using Manopt
        [Y infos] = lowrank_dist_completion_fixedrank(data_train, data_test, Y);
        
        % Some info logging.
        thistime = [infos.time];
        if ~isempty(time)
            thistime = time(end) + thistime;
        end
        time = [time thistime]; %#ok<AGROW>
        cost = [cost [infos.cost]]; %#ok<AGROW>
        if isfield(infos, 'test_error')
            test_error = [test_error [infos.test_error]];
        end
        
        % Evaluate gradient of the convex cost function (i.e. wrt X)
        Z = Y(data_train.rows, :) - Y(data_train.cols,:);
        estimDists = sum(Z.^2,2);
        errors = (estimDists - data_train.entries);
        
        % Dual variable.
        Sy = EIJ * sparse(1:N,1:N,2 * errors / N,N,N) * EIJ';
        
        % Compute smallest algebraic eigenvalue of Sy,
        % this gives us a descent direction for the next rank (v)
        % as well as a way to control progress toward the global
        % optimum (s_min).
        
        % Make eigs silent
        opts.disp = 0;
        
        [v, s_min] = eigs(Sy, 1, 'SA', opts);
        
        % To check whether Y is rank deficient.
        vp = svd(Y);
        
        % Stopping criterion.
        fprintf('>> smin = %.3e, and min(vp) = %.3e\n',s_min,min(vp));
        if (s_min  > -1e-3) || (min(vp) < 1e-3),
            break;
        end
        
        rr = rr + 1; % Update the rank.
        
        if (s_min < -1e-10),
            restartDir = v;
        else
            restartDir = [];
        end
        clear Sy v;
        
    end
    
    fs = 20;
    figure;
    semilogy([cost], '-O','Color','blue','linewidth', 2.0);
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Iterations','FontSize',fs);
    ylabel(ax1,'Cost','FontSize',fs);
    legend('Trust-regions');
    legend 'boxoff';
    box off;
    title('Training on the known distances');
    
    if isfield(infos, 'test_error')
        fs = 20;
        figure;
        semilogy([test_error], '-O','Color','blue','linewidth', 2.0);
        ax1 = gca;
        set(ax1,'FontSize',fs);
        xlabel(ax1,'Iterations','FontSize',fs);
        ylabel(ax1,'Test eror','FontSize',fs);
        legend('Trust-regions');
        legend 'boxoff';
        box off;
        title('Test error on a set different from the training set');
        
    end
    
end



function [Yopt, infos] = lowrank_dist_completion_fixedrank(data_train, data_test, Y_initial)
    
    [n r] = size(Y_initial);
    EIJ = speye(n);
    EIJ = EIJ(:, data_train.rows) - EIJ(:, data_train.cols);
    
    % Create the problem structure
    % quotient YYt (tuned for least square problems) geometry
    problem.M = symfixedrankYYfactory(n,  r);
    
    
    problem.cost = @cost;
    function f = cost(Y)
        xij = EIJ'*Y;
        estimDists = sum(xij.^2,2);
        f = 0.5*mean((estimDists - data_train.entries).^2);
    end
    
    problem.grad = @grad;
    function g = grad(Y)
        N = data_train.nentries;
        xij = EIJ'*Y;
        estimDists = sum(xij.^2,2);
        g = EIJ * sparse(1:N,1:N,2 * (estimDists - data_train.entries) / N, N, N) * xij;
    end
    
    problem.hess = @hess;
    function Hess = hess(Y, eta)
        N = data_train.nentries;
        xij = EIJ'*Y;
        zij = EIJ'*eta;
        estimDists = sum(xij.^2,2);
        crossYZ = 2*sum(xij .* zij,2);
        Hess = (EIJ*sparse(1:N,1:N,2 * (estimDists - data_train.entries) / N,N,N))*zij + (EIJ*sparse(1:N,1:N,2 * crossYZ / N,N,N))*xij;
        Hess = problem.M.proj(Y, Hess);
    end
    
    
    %     % Check numerically whether gradient and Hessian are correct
    %     checkgradient(problem);
    %     drawnow;
    %     pause;
    %     checkhessian(problem);
    %     drawnow;
    %     pause;
    
    
    
    % Ask Manopt to compute the test error at every iteration
    if ~isempty(data_test)
        options.statsfun = @compute_test_error;
        EIJ_test = speye(n);
        EIJ_test = EIJ_test(:, data_test.rows) - EIJ_test(:, data_test.cols);
    end
    function stats = compute_test_error(problem, Y, stats)
        xij = EIJ_test'*Y;
        estimDists_test = sum(xij.^2,2);
        stats.test_error = 0.5*mean((estimDists_test - data_test.entries).^2);
    end
    
    options.stopfun = @mystopfun;
    function stopnow = mystopfun(problem, Y, info, last)
        stopnow = (last >= 3 && (info(last-2).cost - info(last).cost < 1e-3 || abs(info(last-2).cost - info(last).cost)/info(last).cost < 1e-3));
    end
    options.tolgradnorm = 1e-5;
    
    [Yopt, ~, infos] = trustregions(problem, Y_initial, options);
    
    
    
end

