function  test_Karcher_mean_Grass_svrg()
    
    clc; close all; clear;
    
    N   = 10;    
    n   = 500;     
    r   = 5;      
    
    
    % Set manifold
    problem.M = grassmannfactory_modified(n, r);
    
    % Data
    for ii = 1 : N
        data.x{ii} = problem.M.rand();
    end
   
    % Set data
    problem.data = data;
    
    
    % Define problem definitions
    problem.cost = @cost;
    function f = cost(U)
        f = 0;
        for jj = 1 : N
            f = f + 0.5*(problem.M.norm(U, problem.M.log(U,  data.x{jj})))^2;
        end
        f = f/N;
    end
    
    problem.egrad = @egrad;
    function g = egrad(U)
        g = zeros(n, r);
        for jj = 1 : N
            g = g - problem.M.log(U,  data.x{jj});
        end
        g = g/N;
    end
    
    
    problem.egrad_batchsize = @egrad_batchsize;
    function g = egrad_batchsize(U, data_batchsize)
        x_sample_batchsize = data_batchsize.x;
        N_batchsize = length(x_sample_batchsize);
        g = zeros(n, r);
        for jj = 1 : N_batchsize
            g = g - problem.M.log(U,  x_sample_batchsize{jj});
        end
        g = g/N_batchsize;
    end
    
    
    
    %     % Consistency checks
    %     checkgradient(problem)
    %     pause;
    
    
    %
    Uinit = problem.M.rand();
    
    
    % Run SD
    clear options;
    options.maxiter = 500;
    [~, ~, infos_sd, options_sd] = steepestdescent(problem, Uinit, options);
    
    
    % Run SGD
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.update_type='sgd';
    options.maxepoch = 200;
    options.stepsize = 1e0;
    options.stepsize_type = 'decay';
    [~, ~, infos_sgd, options_sgd] = Riemannian_svrg(problem, Uinit, options);
    
    
    % Run SVRG
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.update_type='svrg';
    options.maxepoch = 100;
    options.boost = 1; % Boost on
    options.stepsize = 1e0;
    options.svrg_type = 2;
    options.stepsize_type = 'decay';
    [~, ~, infos_svrg, options_svrg] = Riemannian_svrg(problem, Uinit, options);
    
    
    % Plots
    
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

