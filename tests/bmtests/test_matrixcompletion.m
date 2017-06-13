function test_matrixcompletion()
    
%     cd proposed;
%     addpath(genpath(pwd));
%     cd ..;
    
    clear; clc; close all;
    
    %% Sample data geneteration and problem specifications and options
    n = 100;
    m = 500;
    true_rank = 5;
    over_sampling = 5;
    r = true_rank;
    noiseFac = 1e-6;
    condition_number = 0; % 0 for well-conditioned data and > 0 for ill-conditioned data.
    
    % Print the problem instance information.
    fprintf('Rank %i matrix of size %i times %i and over-sampling = %i\n', true_rank, n, m, over_sampling);
    
    % Generate well-conditioned or ill-conditioned data
    M = over_sampling*true_rank*(m + n -true_rank); % total entries
    
    % The left and right factors which make up our true data matrix Y.
    YL = randn(n, true_rank);
    YR = randn(m, true_rank);
    
    % Condition number
    if condition_number > 0
        YLQ = orth(YL);
        YRQ = orth(YR);
        
        s1 = 1000;
        %     step = 1000; S0 = diag([s1:step:s1+(true_rank-1)*step]*1); % Linear decay
        S0 = s1*diag(logspace(-log10(condition_number),0,true_rank)); % Exponential decay
        
        YL = YLQ*S0;
        YR = YRQ;
        
        fprintf('Creating a matrix with singular values...\n')
        for kk = 1: length(diag(S0));
            fprintf('%s \n', num2str(S0(kk, kk), '%10.5e') );
        end
        singular_vals = svd(YL'*YL);
        condition_number = sqrt(max(singular_vals)/min(singular_vals));
        fprintf('Condition number is %f \n', condition_number);
    end
    
    
    
    % Select a random set of M entries of Y = YL YR'.
    idx = unique(ceil(m*n*rand(1,(10*M))));
    idx = idx(randperm(length(idx)));
    
    [I, J] = ind2sub([n, m], idx(1:M));
    [J, inxs] = sort(J);
    I = I(inxs)';
    
    % Values of Y at the locations indexed by I and J.
    S = sum(YL(I,:).*YR(J,:), 2);
    S_noiseFree = S;
    
    % Add noise.
    noise = noiseFac*max(S)*randn(size(S));
    S = S + noise;
    
    values = sparse(I, J, S, n, m);
    indicator = sparse(I, J, 1, n, m);
    
    % Creat the cells
    samples(m).colnumber = []; % Preallocate memory.
    for k = 1 : m,
        % Pull out the relevant indices and revealed entries for this column
        idx = find(indicator(:, k)); % find known row indices
        values_col = values(idx, k); % the non-zero entries of the column
        
        samples(k).indicator = idx;
        samples(k).values = values_col;
        samples(k).colnumber = k;
    end
    
    
    % Test data
    idx_test = unique(ceil(m*n*rand(1,(10*M))));
    idx_test = idx_test(randperm(length(idx_test)));
    [I_test, J_test] = ind2sub([n, m],idx_test(1:M));
    [J_test, inxs] = sort(J_test); I_test=I_test(inxs)';
    
    % Values of Y at the locations indexed by I and J.
    S_test = sum(YL(I_test,:).*YR(J_test,:), 2);
    values_test = sparse(I_test, J_test, S_test, n, m);
    indicator_test = sparse(I_test, J_test, 1, n, m);
    
    samples_test(m).colnumber = [];
    for k = 1 : m,
        % Pull out the relevant indices and revealed entries for this column
        idx = find(indicator_test(:, k)); % find known row indices
        values_col = values_test(idx, k); % the non-zero entries of the column
        
        samples_test(k).indicator = idx;
        samples_test(k).values = values_col;
        samples_test(k).colnumber = k;
    end
    
    
    %% Call algorithms
    
    % Set manifold and number of cost terms
    problem.M = grassmannfactory(n, r);
    problem.ncostterms = m;
    
    
    % Initialization
    Uinit = problem.M.rand();
    
    
    % For SG/SVRG: no cost/full gradient are needed: only need a partial
    % gradient, which gives the gradient for a selected term of the cost.
    problem.partialegrad = @partialegrad;
         
    % Run SG: no cost/gradient provided. Only partial gradient provided.   
    fprintf('\nRiemannian stochastic gradient algorithm.\n')
    clear options;
    options.verbosity = 2;
    options.batchsize = 10;
    options.maxiter = floor(100*m/10);
    options.checkperiod = floor(5*m/10);
    options.stepsize_init = 1e-3;
    options.stepsize_type = 'decay';

    % Example of how to use statsfunhelper
    metrics.cost_test = @mystatsfun;
    record_cost_grad = true;
    if record_cost_grad
        problem.cost = @cost;
        problem.egrad = @egrad;
        metrics.cost = @(problem, x) getCost(problem, x);
        metrics.gradnorm = @(problem, x) problem.M.norm(x, getGradient(problem, x));
    end
    options.statsfun = statsfunhelper(metrics);
    
    % for stats
    
%     profile clear;
%     profile on;
    [~, info_sg, options_sg] = stochasticgradient(problem, Uinit, options);
%     profile off;
%     profile report;
%     return;    
    
    % Run SVRG
    fprintf('\nRiemannian stochastic variance reduced gradient algorithm.\n')
    clear options;
    options.verbosity = 2;
    options.batchsize = 1;
    options.update_type='svrg';
    options.maxinneriter = 2*m;
    options.maxepoch = 100;
    options.stepsize = 1e-4;
    options.svrg_type = 1;
    options.stepsize_type = 'fix';%'decay' or 'fix' or 'hybrid.
    options.statsfun = statsfunhelper('cost_test', @mystatsfun);
    
    [~, info_svrg, options_svrg] = stochasticvariancereducedgradient(problem, Uinit, options);
       

    % Run SD: cost and gradient provided.
    problem.cost = @cost;
    problem.egrad = @egrad;
    
    % Sanity checks for gradient computation.
    checkgradient(problem);
    
    fprintf('\nRiemannian steepest descent algorithm.\n')
    clear options;
    options.maxiter = 100;
    options.statsfun = statsfunhelper('cost_test', @mystatsfun);
    [~, ~,info_sd, options_sd] = steepestdescent(problem, Uinit, options);
    
    %% Plots
    %%%num_grads_sd = (1:length([infos_sd.cost])) - 1; % N*options_sd.maxiter;
    num_grads_sd = [info_sd.iter];
    %%%num_grads_sg = ceil((options_sg.batchsize*options_sg.savestatsiter*(1 : length([infos_sg.iter])))/m) - 1; 
    num_grads_sg = (options_sg.batchsize*[info_sg.iter])/m;
    %%%next one to be checked went checked SVRG code
    num_grads_svrg = ceil((m + options_svrg.batchsize*options_svrg.maxinneriter)/m)*((1:length([info_svrg.epoch])) - 1); 

    
    % Training loss versus #grads
    fs = 20;
    figure;
    semilogy(num_grads_sd, [info_sd.cost_test], '-O', 'LineWidth', 2, 'MarkerSize', 13);
    hold all;
    semilogy(num_grads_sg, [info_sg.cost_test], '-s', 'LineWidth', 2, 'MarkerSize', 13);
    semilogy(num_grads_svrg, [info_svrg.cost_test], '-*', 'LineWidth', 2, 'MarkerSize', 13);
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Number of batch gradients (equivalent)', 'FontSize',fs);
    ylabel(ax1,'Mean square error on the test set', 'FontSize',fs);
    legend('SD', 'SG', 'SVRG');
    legend 'boxoff';
    box off;
    
    %% Problem definitions
    
    % Cost
    function f = cost(U)
        W = mylsqfit(U, samples);
        f = 0.5*norm(indicator.*(U*W') - values, 'fro')^2;
        f = f/m;
    end
    
    % Gradient
    function g = egrad(U)
        W = mylsqfit(U, samples);
        g = (indicator.*(U*W') - values)*W;
        g = g/m;
    end
    
    % Partial gradient: required for Riemannian SGD.
    function g = partialegrad(U, idx_batchsize)
        g = zeros(n, r);
        m_batchsize = length(idx_batchsize);
        for ii = 1 : m_batchsize
            colnum = idx_batchsize(ii);
            w = mylsqfit(U, samples(colnum));
            indicator_vec = indicator(:, colnum);
            values_vec = values(:, colnum);
            g = g + (indicator_vec.*(U*w') - values_vec)*w;
        end
        g = g/m_batchsize;
        
    end
    
    function f_test = mystatsfun(problem, U)
        W = mylsqfit(U, samples_test);
        f_test = 0.5*norm(indicator_test.*(U*W') - values_test, 'fro')^2;
        f_test = f_test/m;
    end
    
    
    % Compute the least-squares fit.
    function W = mylsqfit(U, currentsamples)
        W = zeros(length(currentsamples), size(U, 2));
        for ii = 1 : length(currentsamples)
            % Pull out the relevant indices and revealed entries for this column
            IDX = currentsamples(ii).indicator;
            values_Omega = currentsamples(ii).values;
            U_Omega = U(IDX,:);
            
            % Solve a simple least squares problem to populate W
            W(ii,:) = (U_Omega\values_Omega)';
        end
    end
    %%
end

