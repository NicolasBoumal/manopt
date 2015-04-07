function [Y, out_infos, out_problem_description] =  lowrank_dist_completion(problem_description)
% Perform low-rank distance matrix completion w/ automatic rank detection.
%
% function Y =  lowrank_dist_completion(problem_description)
% function [Y, infos, out_problem_description] =  lowrank_dist_completion(problem_description)
%
% It implements the ideas of Journee, Bach, Absil and Sepulchre, SIOPT,
% 2010, applied to the problem of low-rank distance matrix completion.
% The details are in the paper "Low-rank optimization for distance matrix completion",
% B. Mishra, G. Meyer, and R. Sepulchre, IEEE CDC, 2011.
%
% Paper link: http://arxiv.org/abs/1304.6663.
%
% Inputs:
% -------
%
% problem_description: The problem structure with the description of the problem.
%
%
% - problem_description.data_train: Data structure for known distances that are used to learn a low-rank model.
%                                   It contains the 3 fields that are shown
%                                   below. An empty "data_train" structure
%                                   will generate a random problem instance
%
%       -- data_train.entries:      A column vector consisting of known
%                                   distances. An empty "data_train.entries"
%                                   field will generate a random
%                                   problem instance..
%
%       -- data_train.rows:         The row position of th corresponding
%                                   distances. An empty "data_train.rows"
%                                   field will generate a random
%                                   problem instance.
%
%       -- data_train.cols:         The column position of th corresponding
%                                   distances. An empty "data_train.cols"
%                                   field will generate a random
%                                   problem instance.
%
%
%
% - problem_description.data_test:  Data structure to compute distances for the "unknown" (to the algorithm) distances.
%                                   It contains the 3 fields that are shown
%                                   below. An empty "data_test" structure
%                                   will not compute the test error.
%
%       -- data_test.entries:       A column vector consisting of "unknown" (to the algorithm)
%                                   distances. An empty "data_test.entries"
%                                   field will not compute the test error.
%       -- data_test.rows:          The row position of th corresponding
%                                   distances. An empty "data_test.rows"
%                                   field will not compute the test error.
%       -- data_test.cols:          The column position of th corresponding
%                                   distances. An empty "data_test.cols"
%                                   field will not compute the test error.
%
%
%
% - problem_description.n:          The number of data points. An empty
%                                   "n", but complete "data_train" structure
%                                   will lead to an error, to avoid
%                                   potential data inconsistency.
%
%
%
% - problem_description.fixedrank_algo: Fixe-rank algorithm. Options are
%                                       'TR' for trust-regions,
%                                       'CG' for conjugate gradients,
%                                       'SD' for steepest descent.
%                                       By default, it is 'TR'.
%
%
%
%
% - problem_description.rank_initial: Starting rank. By default, it is 1.
%
%
%
% - problem_description.rank_max:     Maximum rank. By default, it is equal to
%                                     "problem_description.n".
%
%
%
%
% - problem_description.params:  Structure array containing algorithm
%                                parameters for stopping criteria.
%       -- params.abstolcost:    Tolerance on absolute value of cost.
%                                By default, it is 1e-3.
%
%
%       -- params.reltolcost:    Tolerance on absolute value of cost.
%                                By default, it is 1e-3.
%       -- params.tolgradnorm:   Tolerance on the norm of the gradient.
%                                By default, it is 1e-5.
%       -- params.tolSmin:       Tolerance on smallest eigenvalue of Sy,
%                                the dual variable.
%                                By default, it is 1e-5.
%       -- params.tolrankdeficiency:   Tolerance on the
%                                      smallest singular value of Y.
%                                      By default, it is 1e-3.
%
%
% Outputs:
% --------
%
%   Y:                        n-by-r solution matrix of rank r.
%   out_infos:                Structure array with computed statistics.
%   out_problem_description:  Structure array with used probelm description.
%
%
%
% Please cite the Manopt paper as well as the research paper:
%     @InProceedings{mishra2011dist,
%       Title        = {Low-rank optimization for distance matrix completion},
%       Author       = {Mishra, B. and Meyer, G. and Sepulchre, R.},
%       Booktitle    = {{50th IEEE Conference on Decision and Control}},
%       Year         = {2011},
%       Organization = {{IEEE CDC}}
%     }

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, April 06, 2015.
% Contributors: Nicolas Boumal.
% Change log:

    
    %% Check whether we have complete problem description.
    if ~exist('problem_desription', 'var')...
            || ~isempty(problem_description)...
            || ~all(isfield(problem_description,{'data_train'}) == 1)...
            || ~all(isfield(problem_description.data_train,{'cols', 'rows', 'entries'}) == 1)...
            || isempty(problem_description.data_train.cols)...
            || isempty(problem_description.data_train.rows)...
            || isempty(problem_description.data_train.entries)
        
        %% Generate a random problem instance.
        n = 500; % Number of points
        r = 5; % Embedding dimension
        
        % Fraction of unknown distances
        fractionOfUnknown = 0.8;
        
        
        Yo = randn(n,r); % True embedding
        trueDists = pdist(Yo)'.^2; % True distances
        
        % Add noise (set noise_level = 0 for clean measurements)
        noise_level = 0.01;
        trueDists = trueDists + noise_level * std(trueDists) * randn(size(trueDists));
        
        % Compute all pairs of indices
        H = tril(true(n), -1);
        [I, J] = ind2sub([n, n], find(H(:)));
        clear 'H';
        
        % Train data
        train = false(length(trueDists), 1);
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
        
        % Fixed-rank algorithm
        fixedrank_algo = 'TR'; % Trust-regions.
        
        
        % Rank bounds
        rank_initial = 1; % Starting rank.
        rank_max = n; % Maximum rank.
        
        % Basic parameters that are used in the optimization scheme.
        params.abstolcost = 1e-3;
        params.reltolcost = 1e-3;
        params.tolgradnorm = 1e-5;
        params.tolSmin = -1e-3;
        params.tolrankdeficiency = 1e-3;
        
        
        
        % Collect and output the problem description that are we actual
        % solving.
        out_problem_description.data_train = data_train;
        out_problem_description.data_test = data_test;
        out_problem_description.n = n;
        out_problem_description.fixedrank_algo = 'TR';
        out_problem_description.rank_initial = 1;
        out_problem_description.rank_maximum = n;
        out_problem_description.params = params;
        
    else
        %% Train data
        data_train = problem_description.data_train;
        out_problem_description.data_train = data_train;
        
        
        
        %% Number of data points.
        if isfield(problem_description, 'n')
            n = problem_description.n;
        else
            error('problem_description:field_n_notfound','Error. Number of data points must be given. \n');
        end
        out_problem_description.n = n;
        
        
        
        %% Fixed-rank algorithm to use.
        if isfield(problem_description, 'fixedrank_algo')
            fixedrank_algo = problem_description.fixedrank_algo;
        else
            fixedrank_algo = 'TR';
        end
        out_problem_description.fixedrank_algo = fixedrank_algo;
        
        
        
        %% Initial rank that is given.
        if isfield(problem_description, 'rank_initial') && ~isempty(problem_description.rank_initial)
            rank_initial = problem_description.rank_initial;
        else
            rank_initial = 1;
        end
        out_problem_description.rank_initial = rank_initial;
        
        
        
        %% Maximum rank that is allowed.
        if isfield(problem_description, 'rank_max') && ~isempty(problem_description.rank_max)
            rank_max = problem_description.rank_max;
        else
            rank_max = n;
        end
        out_problem_description.rank_maximum = rank_max;

        
        
        %% Check if a testing dataset is provided.
        if ~isfield(problem_description,{'data_test'})...
                || ~all(isfield(problem_description.data_test,{'cols', 'rows', 'entries'}) == 1)...
                || isempty(problem_description.data_test.cols)...
                || isempty(problem_description.data_test.rows)...
                || isempty(problem_description.data_test.entries)
            data_test = [];
        else % Believe that there is a testing dataset.
            data_test = problem_description.data_test;
        end
        out_problem_description.data_test = data_test;
        
        
        
        %% Parameters
        if isfield(problem_description, 'params')
            params = problem_description.params;
            
            if ~isfield(params,'abstolcost');  params.abstolcost = 1e-3; end
            if ~isfield(params,'reltolcost');  params.reltolcost = 1e-3; end
            if ~isfield(params,'tolgradnorm');  params.tolgradnorm = 1e-5; end
            if ~isfield(params,'tolSmin');  params.tolSmin = -1e-3; end
            if ~isfield(params,'tolrankdeficiency');  params.tolrankdeficiency = 1e-3; end
            
        else
            % Default parameters used in the scheme.
            params.abstolcost = 1e-3;
            params.reltolcost = 1e-3;
            params.tolgradnorm = 1e-5;
            params.tolSmin = -1e-3;
            params.tolrankdeficiency = 1e-3;
            
        end
        out_problem_description.params = params;
        
        
        
    end
    
    
    %% Common 
    N = data_train.nentries; % Number of known distances
    EIJ = speye(n);
    EIJ = EIJ(:, data_train.rows) - EIJ(:, data_train.cols);
    
    rr = rank_initial; % Starting rank.
    Y = randn(n, rr); % Random starting initialization.
    
    time = [];
    cost = [];
    test_error = [];
    rank = [];
    
    
    
    %% Main loop of the rank search algorithm.
    
    while (rr <= rank_max), % When r = n a global min is attained for sure.
        fprintf('>> Rank %d <<\n', rr);
        %% Follow a descent direction to compute an iterate.
        if (rr > rank_initial),
            if isempty(restartDir), % If no restart dir avail. do a random restart.
                
                disp('No restart dir available, random restart is performed');
                Y = randn(n, rr);
                
            else % Perform a simple line-search based on the restart direction.
                
                disp('>> Line-search with restart direction');
                Y(:, rr) = 0; % Append a column of zeroes
                
                Z = Y(data_train.rows, :) - Y(data_train.cols,:);
                estimDists = sum(Z.^2, 2);
                errors = (estimDists - data_train.entries);
                
                grad_Y = EIJ * sparse(1:N,1:N,2 * errors / N,N,N) * Z;
                
                costBefore = mean(errors.^2);
                fprintf('>> Cost before = %f\n',costBefore);
                
                step = N/(n^2); % A very rough estimate of the Lipscitz constant
                for i = 1 : 25, % 25 backtracking to find a descent direction.
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
        
        
        
        
        %% Fixed-rank optimization with Manopt.
        [Y, infos] = lowrank_dist_completion_fixedrank(fixedrank_algo, data_train, data_test, Y, params);
        
        
        
        %% Some info logging.
        thistime = [infos.time];
        if ~isempty(time)
            thistime = time(end) + thistime;
        end
        time = [time thistime]; %#ok<AGROW>
        cost = [cost [infos.cost]]; %#ok<AGROW>
        rank = [rank [infos.rank]];
        
        if isfield(infos, 'test_error')
            test_error = [test_error [infos.test_error]]; %#ok<AGROW>
        end
        
        
        
        %% Evaluate gradient of the convex cost function (i.e. wrt X).
        Z = Y(data_train.rows, :) - Y(data_train.cols,:);
        estimDists = sum(Z.^2,2);
        errors = (estimDists - data_train.entries);
        
        
        
        %% Dual variable and its minimum eigenvalue that is used to guarantee convergence.
        Sy = EIJ * sparse(1:N,1:N,2 * errors / N,N,N) * EIJ';
        
        
        % Compute smallest algebraic eigenvalue of Sy,
        % this gives us a descent direction for the next rank (v)
        % as well as a way to control progress toward the global
        % optimum (s_min).
        
        % Make eigs silent.
        opts.disp = 0;
        opts.issym = true;
        [v, s_min] = eigs(Sy, 1, 'SA', opts);
        
        
        
        %% Check whether Y is rank deficient.
        vp = svd(Y);
        
        % Stopping criterion.
        fprintf('>> smin = %.3e, and min(vp) = %.3e\n',s_min,min(vp));
        if (s_min  > params.tolSmin) || (min(vp) < params.tolrankdeficiency),
            break;
        end
        
        
        
        %% Update rank
        rr = rr + 1; % Update the rank.
        
        
        
        %% Compute a descent direction.
        if (s_min < -1e-10),
            restartDir = v;
        else
            restartDir = [];
        end
        clear Sy v;
        
        
        
    end
    
    
    %% Collect the relevant statistics that we have.
    out_infos.time = time;
    out_infos.cost = cost;
    out_infos.rank = rank;
    out_infos.test_error = test_error;
    
    
    %% Time for plots.
    fs = 20;
    figure;
    semilogy(cost, '-O', 'Color', 'blue', 'linewidth', 2.0);
    ax1 = gca;
    set(ax1, 'FontSize', fs);
    xlabel(ax1, 'Iterations', 'FontSize', fs);
    ylabel(ax1, 'Cost', 'FontSize', fs);
    legend('Trust-regions');
    legend 'boxoff';
    box off;
    title('Training on the known distances');
    
    if isfield(infos, 'test_error')
        fs = 20;
        figure;
        semilogy(test_error, '-O', 'Color', 'blue', 'linewidth', 2.0);
        ax1 = gca;
        set(ax1, 'FontSize', fs);
        xlabel(ax1, 'Iterations', 'FontSize', fs);
        ylabel(ax1, 'Test error', 'FontSize', fs);
        legend('Trust-regions');
        legend 'boxoff';
        box off;
        title('Test error on a set of distances different from the training set');
        
    end
    
end



function [Yopt, infos] = lowrank_dist_completion_fixedrank(fixedrank_algo, data_train, data_test, Y_initial, params)
    
    %% Common 
    [n, r] = size(Y_initial);
    EIJ = speye(n);
    EIJ = EIJ(:, data_train.rows) - EIJ(:, data_train.cols);
    
    
    
    %% Create the problem structure
    problem.M = symfixedrankYYfactory(n,  r);
    
    
    
    %% Cost evaluation
    problem.cost = @cost;
    function [f, store] = cost(Y, store)
        if ~isfield(store, 'xij')
            store.xij = EIJ'*Y;
        end
        xij = store.xij;
        estimDists = sum(xij.^2,2);
        f = 0.5*mean((estimDists - data_train.entries).^2);
    end
    
    
    
    %% Gradient evaluation.
    problem.grad = @grad;
    function [g, store] = grad(Y, store)
        N = data_train.nentries;
        if ~isfield(store, 'xij')
            store.xij = EIJ'*Y;
        end
        xij = store.xij;
        estimDists = sum(xij.^2,2);
        g = EIJ * sparse(1:N,1:N,2 * (estimDists - data_train.entries) / N, N, N) * xij;
    end
    
    
    
    %% Hessian evaluation.
    problem.hess = @hess;
    function [Hess, store] = hess(Y, eta, store)
        N = data_train.nentries;
        if ~isfield(store, 'xij')
            store.xij = EIJ'*Y;
        end
        xij = store.xij;
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
    
    
    
    %% When asked, ask Manopt to compute the test error at every iteration.
    if ~isempty(data_test)
        options.statsfun = @compute_test_error;
        EIJ_test = speye(n);
        EIJ_test = EIJ_test(:, data_test.rows) - EIJ_test(:, data_test.cols);
    end
    function stats = compute_test_error(problem, Y, stats) %#ok<INUSL>
        xij = EIJ_test'*Y;
        estimDists_test = sum(xij.^2,2);
        stats.test_error = 0.5*mean((estimDists_test - data_test.entries).^2);
        stats.rank = r;
    end
    
    
    
    %% Stopping criteria
    
    options.stopfun = @mystopfun;
    function stopnow = mystopfun(problem, Y, info, last) %#ok<INUSL>
        stopnow = (last >= 3 && (info(last-2).cost - info(last).cost < params.abstolcost || abs(info(last-2).cost - info(last).cost)/info(last).cost < params.reltolcost));
    end
    options.tolgradnorm = params.tolgradnorm;
    
    
    
    %% Call the appropriate algorithm.
    if strcmp(fixedrank_algo, 'TR'),
        [Yopt, ~, infos] = trustregions(problem, Y_initial, options);
    elseif strcmp(fixedrank_algo, 'CG'),
        [Yopt, ~, infos] = conjugategradient(problem, Y_initial, options);
    elseif strcmp(fixedrank_algo, 'SD'),
        [Yopt, ~, infos] = steepestdescent(problem, Y_initial, options);
    else % By default
        [Yopt, ~, infos] = trustregions(problem, Y_initial, options);
    end
    
    
    
end

