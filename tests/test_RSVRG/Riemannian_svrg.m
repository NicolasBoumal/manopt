function [x, cost, info, options] = Riemannian_svrg(problem, x_init, options)
% The Riemannian SVRG/SGD algorithms. 
%
% function [x, cost, info, options] = Riemannian_svrg(problem)
% function [x, cost, info, options] = Riemannian_svrg(problem, x_init)
% function [x, cost, info, options] = Riemannian_svrg(problem, x_init, options)
% function [x, cost, info, options] = Riemannian_svrg(problem, [], options)
% 
% Apply the Riemannian SVRG/SGD algorithm to the problem defined
% in the problem structure, starting at x_init if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x_init as [] (the empty matrix).
% 
% The solver mimics other solvers of Manopt with two additonal input
% requirements: problem.data and problem.egrad_batchsize. 
%
% problem.samples is a structure array to the all the data samples. 
% For example, problem.samples should be a struct array of N samples.
%
% problem.egrad_batchsize takes input a current point of the manifold and
% samples_batchsize that contains the 'samples' samples only for the current
% batchsize.
%
% Some of the options of the solver are specifict to this file. Please have
% a look below.
%
% The solver is based on the paper by
% H. Kasai, H. Sato, and B. Mishra,
% "Riemannian stochastic variance reduced gradient on Grassmann manifold,"
% Technical report, arXiv preprint arXiv:1605.07367, 2016.


    
% Original authors: Bamdev Mishra <bamdevm@gmail.com>,
%                   Hiroyuki Kasai <kasai@is.uec.ac.jp>, and
%                   Hiroyuki Sato <hsato@ms.kagu.tus.ac.jp>, 22 April 2016.
    
    % Initialization
    if isempty(x_init)
        x_init =  problem.M.rand();
    end
    x = x_init;
    
    % Data
    samples = problem.samples; % struct array .
    
    
    % Total number of samples
    N = length(samples);
    
    % Extract options
    if ~isfield(options, 'maxepoch'); options.maxepoch = 100; end; % Maximum number of epochs.
    if ~isfield(options, 'maxinneriter'); options.maxinneriter = 5*N; end; % Maximum number of sampling per epoch.
    if ~isfield(options, 'stepsize'); options.stepsize = 0.1; end % Initial stepsize guess.
    if ~isfield(options, 'stepsize_type'); options.stepsize_type = 'decay'; end % Stepsize type. Other possibilities are 'fix' and 'hybrid'.
    if ~isfield(options, 'stepsize_lambda'); options.stepsize_lambda = 0.1; end % lambda is a weighting factor while using stepsize_typ='decay'.
    if ~isfield(options, 'tolgradnorm'); options.tolgradnorm = 1.0e-6; end % Batch grad norm tolerance.
    if ~isfield(options, 'batchsize'); options.batchsize = 1; end % Batchsize.
    if ~isfield(options, 'verbosity'); options.verbosity = 0; end % Output verbosity. Other options are 1 and 2.
    if ~isfield(options, 'transport'); options.transport = 'vector';end % Other option is 'parallel', but this should be defined in the manifold factory.
    if ~isfield(options, 'boost'); options.boost = false;  end % True: do a normal SGD at the first epoch when SVRG.
    if ~isfield(options, 'update_type'); options.update_type = 'svrg';  end % Update type. Other possibility is 'sgd', which is the standard SGD.
    if ~isfield(options, 'store_innerinfo'); options.store_innerinfo = false; end % Store information at each update. High memory requirements. Only to be used for debugging.
    if ~isfield(options, 'statsfun'); options.statsfun = []; end % A function handle that gets exectued at every epoch.
    if ~isfield(options, 'svrg_type'); options.svrg_type = 1; end % To implement both the options that are used to define x0.
    
    
    stepsize0 = options.stepsize;
    batchsize = options.batchsize;
    
    
    % Total number of batches
    totalbatches = ceil(options.maxinneriter/batchsize);
    
    
    
    % Computations at initialization
    cost = problem.cost(x);
    egrad = problem.egrad(x);
    rgrad = problem.M.proj(x, egrad);
    gradnorm = problem.M.norm(x, rgrad);
    
    % Save stats in a struct array info, and preallocate.
    epoch = 0;
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxepoch+1)).epoch = [];
    info(min(10000, options.maxepoch+1)).cost = [];
    info(min(10000, options.maxepoch+1)).time = [];
    info(min(10000, options.maxepoch+1)).gradnorm = [];
    
    % Initialize innerinfo
    iter = 0;
    inneriter = 0;
    if options.store_innerinfo
        innerstats = saveinnerstats();
        innerinfo(1) = innerstats;
        info(1).innerinfo = innerinfo;
        innerinfo(min(10000, totalbatches+1)).inneriter = [];
    end
    
    
    if options.verbosity > 0
        fprintf('-------------------------------------------------------\n');
        fprintf('R-%s:  epoch\t               cost val\t    grad. norm\t stepsize\n', options.update_type);
        fprintf('R-%s:  %5d\t%+.16e\t%.8e\t%.8e\n', options.update_type, 0, cost, gradnorm,stepsize0);
        
        
        if options.verbosity > 1
            fprintf('             inneriter\t               cost val\t    grad. norm\n');
        end
    end
    
    
    x0 = x;
    rgrad0 = rgrad;
    toggle = 0; % To check boosting.
    % Main loop over epoch.
    for epoch = 1 : options.maxepoch
        
        % Draw the samples with replacement.
        perm_idx = randi(N, 1, options.maxinneriter);
        samples_perm = samples(perm_idx);
        
        
        % Check if boost is required for svrg
        if strcmp(options.update_type, 'svrg') && options.boost && epoch == 1
            options.update_type = 'sgd';
            toggle = 1;
        end
        
        if strcmp(options.update_type, 'svrg') && options.svrg_type == 2
            update_instance = randi(totalbatches, 1) - 1; % pick a number uniformly between 0 to m - 1.
            if update_instance == 0
                Usave = x0;
                rgradsave = rgrad0;
            end
        end
        
        
        elapsed_time = 0;
        % Per epoch: main loop over samples.
        for inneriter = 1 : totalbatches
            
            % Set start time
            start_time = tic;
            
            % Pick a sample of size batchsize
            start_index = (inneriter - 1)* batchsize + 1;
            end_index = min(inneriter * batchsize, options.maxinneriter);
            
            samples_batchsize = samples_perm(start_index : end_index); % An array.
            
            
            % Compute gradient on this batch.
            egrad_batchsize = problem.egrad_batchsize(x, samples_batchsize); % BM: per batchsize
            rgrad_batchsize = problem.M.egrad2rgrad(x, egrad_batchsize); % BM
            
            % Update stepsize
            if strcmp(options.stepsize_type, 'decay')
                stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
           
            elseif strcmp(options.stepsize_type, 'fix')
                stepsize = stepsize0; % Fixed stepsize.
                
            elseif strcmp(options.stepsize_type, 'hybrid')
                if epoch < 5 % Decay stepsize only for the initial few epochs.
                    stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
                end
                
            else
                error(['Unknown options.stepsize_type. ' ...
                    'Should be fix or decay.']);
            end
            
            
            % Update rgrad_batchsize
            
            if strcmp(options.update_type, 'svrg')
                
                % Caclculate transported full batch gradient from x0 to x.
                if strcmp(options.transport, 'parallel')
                    % Logarithm map
                    logmapU0toU = problem.M.log(x0, x);
                    
                    % Parallel translate from x0 to x.
                    rgrad_transported = problem.M.paratransp(x0, logmapU0toU, rgrad0); % BM: this is in _mod file.
                    
                else 
                    rgrad_transported = problem.M.transp(x0, x, rgrad0); % Vector transport.
                end
                
                % Caclculate egrad_batchsize and rgrad_batchsize at x0
                egrad0_batchsize = problem.egrad_batchsize(x0, samples_batchsize);
                rgrad0_batchsize = problem.M.egrad2rgrad(x0, egrad0_batchsize);
                
                
                % Caclculate transported rgrad_batchsize from x0 to x
                if strcmp(options.transport, 'parallel')
                    % parallel translate from x0 to x
                    rgrad0_batchsize_transported = problem.M.paratransp(x0, logmapU0toU, rgrad0_batchsize);
                    
                else
                    rgrad0_batchsize_transported = problem.M.transp(x0, x, rgrad0_batchsize); % Vector transport.
                    
                end
                
                % Update rgrad_batchsize to reduce variance by
                % taking a linear combination with old gradients.
                rgrad_batchsize = problem.M.lincomb(x, 1, rgrad_transported, 1, rgrad_batchsize);
                rgrad_batchsize = problem.M.lincomb(x, 1, rgrad_batchsize, -1, rgrad0_batchsize_transported);
                
                
            elseif strcmp(options.update_type, 'sgd')
                % Do nothing
                
            else
                error(['Unknown options.update_type. ' ...
                    'Should be svrg or sgd.']);
                
            end
            
            % Update x
            x = problem.M.exp(x, rgrad_batchsize, -stepsize);
            
            % Elapsed time
            elapsed_time = elapsed_time + toc(start_time);
            
            iter = iter + 1; % Total number updates.
            
            if strcmp(options.update_type, 'svrg') && options.svrg_type == 2 && inneriter == update_instance
                Usave = x;
                egrad = problem.egrad(x);
                rgrad = problem.M.egrad2rgrad(x, egrad);
                rgradsave = rgrad;
            end
            
            
            if options.store_innerinfo
                cost = problem.cost(x);
                gradnorm = problem.M.norm(x, rgrad_batchsize);
                innerstats = saveinnerstats();
                innerinfo(inneriter) = innerstats;
                if options.verbosity > 1
                    fprintf('R-%: %5d (%5d)\t%+.16e\t%.8e\t%.8e\n', options.update_type, inneriter, epoch, cost, gradnorm, stepsize);
                end
            end
            
        end
        
        % Calculate cost, rgrad, and gradnorm
        cost = problem.cost(x);
        if strcmp(options.update_type, 'svrg') && options.svrg_type == 2
            x0 = Usave;
            rgrad0 = rgradsave;
        else
            if strcmp(options.update_type, 'svrg')
                tsvrg = tic;
            end
            egrad = problem.egrad(x);
            rgrad = problem.M.egrad2rgrad(x, egrad);
            if strcmp(options.update_type, 'svrg')
                elapsed_time = elapsed_time + toc(tsvrg);
            end
            x0 = x;
            rgrad0 = rgrad;
        end
        
        gradnorm = problem.M.norm(x, rgrad);
        
        % Log statistics for freshly executed iteration
        stats = savestats();
        if options.store_innerinfo
            stats.innerinfo = innerinfo;
        end
        info(epoch+1)= stats;
        if options.store_innerinfo
            info(epoch+1).innerinfo = innerinfo;
        end
        
        % Reset if boosting used already.
        if toggle == 1
            options.update_type = 'svrg';
        end
        
        % Print output
        if options.verbosity > 0
            fprintf('R-%s:  %5d\t%+.16e\t%.8e\t%.8e\n',options.update_type, epoch, cost, gradnorm, stepsize);
        end
        
        % Stopping criteria
        if gradnorm  <= options.tolgradnorm
            if options.verbosity > 0
                fprintf('Norm of gradient smaller than %g.\n',options.tolgradnorm);
            end
            break;
        end
    end
    
    info = info(1:epoch+1);
    
    
    % Save the stats per epoch.
    function stats = savestats()
        stats.epoch = epoch;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        if epoch == 0
            stats.time = 0;
        else
            stats.time = info(epoch).time + elapsed_time;
        end
        
        if  ~isempty(options.statsfun)
            stats = options.statsfun(problem, x, stats);
        end
    end
    
    % Save the stats per iteration.
    function innerstats = saveinnerstats()
        innerstats.inneriter = inneriter;
        if inneriter == 0
            innerstats.cost = NaN;
            innerstats.gradnorm = NaN;
            innerstats.time = 0;
        else
            innerstats.cost = cost;
            innerstats.gradnorm = gradnorm;
            if inneriter == 1
                innerstats.time = elapsed_time;
            else
                innerstats.time = innerinfo(inneriter-1).time + elapsed_time;
            end
        end
        
    end
    
    
end


