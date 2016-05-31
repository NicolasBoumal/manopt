function  test_PCA_Grass_svrg()
    
    clc; close all; clear;
    
    % Load input data
    NumSamples  = 500;     % {500, 1000}
    dimValue    = 20;      % {100}
    rankvalue   = 5;       % {5}
    
    
    input_filename = sprintf('./data/pca/pca_samples_%d_d_%d_r_%d.mat', NumSamples, dimValue, rankvalue);
    input_data = load(input_filename);
    
    d = input_data.d;
    r = input_data.r;
    x_sample = input_data.x_sample;
    
    
    
    N = size(x_sample, 2); % Total number of samples.

    
    % Set manifold
    problem.M = grassmannfactory(d, r);
    
    
    
    
    % Data for input as cell
    data.x = mat2cell(x_sample, dimValue, ones( NumSamples, 1)); % BM: okay.
    
    
    
    %Set data
    problem.data = data;
    
    
    
    
    % Define problem definitions
    problem.cost = @cost;
    function f = cost(U)
        f = -0.5*norm(U'*x_sample, 'fro')^2;
        f = f/N;
    end
    
    problem.egrad = @egrad;
    function g = egrad(U)
        g = - x_sample*(x_sample'*U);
        g = g/N;
    end
    
    
    problem.egrad_batchsize = @egrad_batchsize;
    function g = egrad_batchsize(U, data_batchsize)
        x_sample_batchsize = cell2mat(data_batchsize.x);        
        N_batchsize = size(x_sample_batchsize, 2);
        g = - x_sample_batchsize*(x_sample_batchsize'*U);
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
    options.stepsize_type = 'hybrid';
    [~, ~, infos_sgd, options_sgd] = Riemannian_svrg(problem, Uinit, options);
    
    
    % Run SVRG
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.update_type='svrg';
    options.maxepochs = 100;
    options.stepsize = 1e-2;
    options.svrg_type = 2;
    options.stepsize_type = 'hybrid';
    [~, ~, infos_svrg, options_svrg] = Riemannian_svrg(problem, Uinit, options);
    
    
    %% Plots
    x_star = input_data.x_star;
    f_sol = problem.cost(x_star);
    
    
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

