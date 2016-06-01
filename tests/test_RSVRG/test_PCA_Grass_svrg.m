function  test_PCA_Grass_svrg()
    
    clc; close all; clear;
    
    N = 500;
    d = 20;
    r = 5;
    samples_mat = randn(d, N);

    
    % Set manifold
    problem.M = grassmannfactory(d, r);
    
    
    
    % Data for input as cell
    samples = mat2cell(samples_mat, d, ones(N, 1)); % BM: okay.
    
    
    
    %Set data
    problem.samples = samples;
    
    
    
    
    % Define problem definitions
    problem.cost = @cost;
    function f = cost(U)
        f = -0.5*norm(U'*samples_mat, 'fro')^2;
        f = f/N;
    end
    
    problem.egrad = @egrad;
    function g = egrad(U)
        g = - samples_mat*(samples_mat'*U);
        g = g/N;
    end
    
    
    problem.egrad_batchsize = @egrad_batchsize;
    function g = egrad_batchsize(U, samples_batchsize)
        samples_batchsize = cell2mat(samples_batchsize);        
        N_batchsize = size(samples_batchsize, 2);
        g = - samples_batchsize*(samples_batchsize'*U);
        g = g/N_batchsize;
    end
    
    
    
    %     % Consistency checks
    %     checkgradient(problem)
    %     pause;
    
    
    % Intialiaization
    Uinit = problem.M.rand();
    
    
    % Run SD
    clear options;
    options.maxiter = 100;
    [~, ~, infos_sd, options_sd] = steepestdescent(problem, Uinit, options);
    
    
    
    % Run SGD
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.update_type='sgd';
    options.maxepochs = 100;
    options.stepsize = 1e-3;
    options.stepsize_type = 'decay';
    [~, ~, infos_sgd, options_sgd] = Riemannian_svrg(problem, Uinit, options);
    
    
    % Run SVRG
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.update_type='svrg';
    options.maxepochs = 100;
    options.stepsize = 1e-1;
    options.svrg_type = 2;
    options.stepsize_type = 'hybrid';
    [~, ~, infos_svrg, options_svrg] = Riemannian_svrg(problem, Uinit, options);
    
    
    
    %% Plots
    [U_star, ~, ~] = svds(samples_mat, r);
    f_sol = problem.cost(U_star);
    
    
    error_sd = abs([infos_sd.cost] - f_sol);
    error_sgd = abs([infos_sgd.cost] - f_sol);
    error_svrg = abs([infos_svrg.cost] - f_sol);
    
    
    num_grads_sd = (1:length([infos_sd.cost])) - 1; % N*options_sd.maxiter;
    num_grads_sgd = ceil(options_sgd.maxinneriter/N)*((1:length([infos_sgd.cost])) - 1); % options.maxepoch*(options_sgd.maxinneriter);
    num_grads_svrg = ceil((N + options_svrg.maxinneriter)/N)*((1:length([infos_svrg.cost])) - 1); %options.maxepoch*(N + options_svrg.maxinneriter); % Per epoch we compute equivalent of 2 batch grads.
    
  
    
    % Training loss versus #grads
    fs = 20;
    figure;
    plot(num_grads_sd, [infos_sd.cost],'-O','Color','m','LineWidth',2, 'MarkerSize',13);
    hold on;
    plot(num_grads_sgd, [infos_sgd.cost],'-s','Color','r','LineWidth',2, 'MarkerSize',13);
    plot(num_grads_svrg, [infos_svrg.cost],'-*','Color','b','LineWidth',2, 'MarkerSize',13);
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Number of full gradients','FontSize',fs);
    ylabel(ax1,'Mean square error on training set','FontSize',fs);
    legend('SD', 'SGD','SVRG');
    legend 'boxoff';
    box off;

    
    % Training loss - optimum versus #grads
    fs = 20;
    figure;
    semilogy(num_grads_sd, error_sd,'-O','Color','m','LineWidth',2, 'MarkerSize',13);
    hold on;
    semilogy(num_grads_sgd, error_sgd,'-s','Color','r','LineWidth',2, 'MarkerSize',13);
    semilogy(num_grads_svrg, error_svrg,'-*','Color','b','LineWidth',2, 'MarkerSize',13);
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Number of batch gradients','FontSize',fs);
    ylabel(ax1,'Training loss - optimum','FontSize',fs);
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

