function  test_MC_Grass_svrg()
    
    clear; clc; close all;
    
    
    %% Sample data geneteration
    
    % Problem specifications and options
    n = 100;
    m = 500;
    true_rank = 5;
    over_sampling = 5;
    
    r = true_rank;
    noiseFac = 1e-6;
    condition_number = 0; % 0 for well-conditioned data; > 0 for ill-conditioned data.
    
  
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
    
    [I, J] = ind2sub([n, m],idx(1:M));
    [J, inxs] = sort(J); I=I(inxs)';
    
    % Values of Y at the locations indexed by I and J.
    S = sum(YL(I,:).*YR(J,:), 2);
    S_noiseFree = S;
    
    % Add noise.
    noise = noiseFac*max(S)*randn(size(S));
    S = S + noise;
    
    
    values = sparse(I, J, S, n, m);
    indicator = sparse(I, J, 1, n, m);
    
    
    
    % Creat the cells
    data.x(m).colnumber = []; % Preallocate memory.
    for k = 1 : m,
        % Pull out the relevant indices and revealed entries for this column
        idx = find(indicator(:, k)); % find known row indices
        values_col = values(idx, k); % the non-zero entries of the column
        
        data.x(k).indicator = idx;
        data.x(k).values = values_col;
        data.x(k).colnumber = k;
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
    
    data_test.x(m).colnumber = [];
    for k = 1 : m,
        % Pull out the relevant indices and revealed entries for this column
        idx = find(indicator_test(:, k)); % find known row indices
        values_col = values_test(idx, k); % the non-zero entries of the column
        
        data_test.x(k).indicator = idx;
        data_test.x(k).values = values_col;
        data_test.x(k).colnumber = k;
    end
    
    
    
    %% Set manifold
    problem.M = grassmannfactory(n, r);
    
    
    
    %% Set data
    problem.data = data;
   
    
    %% Define problem definitions
    problem.cost = @cost;
    function f = cost(U)
        W = mylsqfit(U, data.x);
        f = 0.5*norm(indicator.*(U*W') - values, 'fro')^2;
        f = f/m;
    end
    
    problem.egrad = @egrad;
    function g = egrad(U)
        W = mylsqfit(U, data.x);
        g = (indicator.*(U*W') - values)*W;
        g = g/m;
    end
    
    
    problem.egrad_batchsize = @egrad_batchsize;
    function g = egrad_batchsize(U, data_batchsize)
        g = zeros(n, r);
        m_batchsize = length([data_batchsize.x.colnumber]);
        for ii = 1 : m_batchsize
            colnum = data_batchsize.x(ii).colnumber;
            w = mylsqfit(U, data_batchsize.x(ii));
            indicator_vec = indicator(:, colnum);
            values_vec = values(:, colnum);
            g = g + (indicator_vec.*(U*w') - values_vec)*w;
        end
        g = g/m_batchsize;
        
    end
    
    
    %     function stats = mystatsfun(problem, U, stats)
    %         W = mylsqfit(U, data_test.x);
    %         f_test = 0.5*norm(indicator_test.*(U*W') - values_test, 'fro')^2;
    %         f_test = f_test/m;
    %         stats.cost_test = f_test;
    %     end
    
    
    %     % Consistency checks
    %     checkgradient(problem)
    %     pause;
    
    
    % Initialization
    Uinit = problem.M.rand();
    
    
    % Run SD
    clear options;
    options.maxiter = 100;
    %     options.statsfun = @mystatsfun;
    
    [~, ~,infos_sd, options_sd] = steepestdescent(problem, Uinit, options);
    
    
    % Run SGD
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.update_type='sgd';
    options.maxepochs = 100;
    options.stepsize = 1e-3;
    options.stepsize_type = 'decay';
    %     options.statsfun = @mystatsfun;
    
    [~, ~, infos_sgd, options_sgd] = Riemannian_svrg(problem, Uinit, options);
    
    
    % Run SVRG
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.update_type='svrg';
    options.maxepochs = 100;
    options.stepsize = 9e-4;
    %     options.statsfun = @mystatsfun;
    options.svrg_type = 1;
    options.stepsize_type = 'decay'; % 'fix' or 'hybrid.
    
    [~, ~, infos_svrg, options_svrg] = Riemannian_svrg(problem, Uinit, options);
    
    
    
    %% Plots
    num_grads_sd = (1:length([infos_sd.cost])) - 1; % N*options_sd.maxiter;
    num_grads_sgd = ceil(options_sgd.maxinneriter/m)*((1:length([infos_sgd.cost])) - 1); % options.maxepoch*(options_sgd.maxinneriter);
    num_grads_svrg = ceil((m + options_svrg.maxinneriter)/m)*((1:length([infos_svrg.cost])) - 1); %options.maxepoch*(N + options_svrg.maxinneriter); % Per epoch we compute equivalent of 2 batch grads.
    
    
    
    % Training loss versus #grads
    fs = 20;
    figure;
    semilogy(num_grads_sd, [infos_sd.cost],'-O','Color','m','LineWidth',2, 'MarkerSize',13);
    hold on;
    semilogy(num_grads_sgd, [infos_sgd.cost],'-s','Color','r','LineWidth',2, 'MarkerSize',13);
    semilogy(num_grads_svrg, [infos_svrg.cost],'-*','Color','b','LineWidth',2, 'MarkerSize',13);
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Number of batch gradients','FontSize',fs);
    ylabel(ax1,'Mean square error on training set','FontSize',fs);
    legend('SD', 'SGD','SVRG');
    legend 'boxoff';
    box off;
    
    
    
    
    
    % Gradient norm versus #grads
    fs = 20;
    figure;
    semilogy(num_grads_sd, [infos_sd.gradnorm],'-O','Color','m','LineWidth',2, 'MarkerSize',13);
    hold on;
    semilogy(num_grads_sgd, [infos_sgd.gradnorm],'-s','Color','r','LineWidth',2, 'MarkerSize',13);
    semilogy(num_grads_svrg, [infos_svrg.gradnorm],'-*','Color','b','LineWidth',2, 'MarkerSize',13);
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Number of batch gradients','FontSize',fs);
    ylabel(ax1,'Norm of gradient','FontSize',fs);
    legend('SD', 'SGD','SVRG');
    legend 'boxoff';
    box off;
    
    
end

