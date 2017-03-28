function [x, info, options] = stochasticvariancereducedgradient(problem, x, options)
% Stochastic variance reduced gradient (SVRG) min. algorithm for Manopt.
%
% function [x, info, options] = stochasticvariancereducedgradient(problem)
% function [x, info, options] = stochasticvariancereducedgradient(problem, x)
% function [x, info, options] = stochasticvariancereducedgradient(problem, x, options)
% function [x, info, options] = stochasticvariancereducedgradient(problem, [], options)
%
% Apply the stochasticvariancereducedgradient algorithm to the
% problem defined in the problem structure, starting at x if it is provided
% (otherwise, at a random point on the manifold).
% To specify options whilst not specifying an initial guess, give x as [] (the empty matrix).
%
% The solver mimics other solvers of Manopt with two additonal input
% requirements: problem.ncostterms and problem.partialegrad.
%
% problem.ncostterms has the number of samples, e.g., problem.ncostterms samples.
%
% problem.partialegrad takes input a current point of the manifold and
% index of batchsize.
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
    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetPartialGradient(problem)
        warning('manopt:getPartialGradient', ...
            'No partial gradient provided. The algorithm will likely abort.');
    end
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    
    % Set local defaults
    localdefaults.maxepoch = 100; % Maximum number of epochs.
    localdefaults.maxinneriter = 5*problem.ncostterms; % Maximum number of sampling per epoch.
    localdefaults.stepsize = 0.1; % Initial stepsize guess.
    localdefaults.stepsize_type = 'decay'; % Stepsize type. Other possibilities are 'fix' and 'hybrid'.
    localdefaults.stepsize_lambda = 0.1; % lambda is a weighting factor while using stepsize_typ='decay'.
    localdefaults.tolgradnorm = 1.0e-6; % Batch grad norm tolerance.
    localdefaults.batchsize = 1; % Batchsize.
    localdefaults.verbosity = 0; % Output verbosity. Other localdefaults are 1 and 2.
    localdefaults.boost = false; % True: do a normal SGD at the first epoch when SVRG.
    localdefaults.update_type = 'svrg';   % Update type. Other possibility is 'sgd', which is the standard SGD.
    localdefaults.saveinnerstats = false; % Store information at each update. High memory requirements. Only to be used for debugging.
    localdefaults.svrg_type = 1;  % To implement both the localdefaults that are used to define x0.
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    
    stepsize0 = options.stepsize;
    batchsize = options.batchsize;
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x
    if canGetCost(problem)
        [cost, grad] = getCostGrad(problem, x, storedb, key);
        gradnorm = problem.M.norm(x, grad);
    elseif canGetGradient(problem)
        grad = getGradient(problem, x, storedb, key);
        gradnorm = problem.M.norm(x, grad);
    end
    
    % Save stats in a struct array info, and preallocate.
    epoch = 0;
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxepoch+1)).epoch = [];
    info(min(10000, options.maxepoch+1)).time = [];
    if canGetCost(problem)
        info(min(10000, options.maxepoch+1)).cost = [];
    end
    if canGetGradient(problem)
        info(min(10000, options.maxepoch+1)).gradnorm = [];
    end
    
    % Initialize innerinfo
    iter = 0;
    inneriter = 0;
    if options.saveinnerstats
        innerstats = saveinnerstats();
        innerinfo(1) = innerstats;
        info(1).innerinfo = innerinfo;
        innerinfo(min(10000, ceil(options.maxinneriter)+1)).inneriter = [];
    end
    
    if options.verbosity > 0 && canGetCost(problem) && canGetGradient(problem)
        fprintf('-------------------------------------------------------\n');
        fprintf('%s:    epoch\t               cost val\t    grad. norm\t      stepsize\n', options.update_type);
        fprintf('%s:    %5d\t%+.16e\t%.8e\t%.8e\n', options.update_type, 0, cost, gradnorm,stepsize0);
        if options.saveinnerstats && options.verbosity > 1
            fprintf('               inneriter                       cost val partialgrad. norm     stepsize\n');
        end
    elseif options.saveinnerstats && options.verbosity > 1
        fprintf('%s:    epoch inneriter     partialgrad. norm\t      stepsize\n',options.update_type);
    end
    
    x0 = x;
    grad0 = grad;
    toggle = 0; % To check boosting.
   
    % Main loop over epoch.
    for epoch = 1 : options.maxepoch
        
        % Check if boost is required for svrg
        if strcmp(options.update_type, 'svrg') && options.boost && epoch == 1
            toggle = 1;
        end
        
        if strcmp(options.update_type, 'svrg') && options.svrg_type == 2
            update_instance = randi(options.maxinneriter, 1) - 1; % pick a number uniformly between 0 to m - 1.
            if update_instance == 0
                xsave = x0;
                gradsave = grad0;
            end
        end
        
        % Draw the samples with replacement.
        perm_idx = randi(problem.ncostterms, 1, batchsize*options.maxinneriter);
        
        
        elapsed_time = 0;
        
        % Per epoch: main loop over samples.
        for inneriter = 1 : options.maxinneriter
            
            % Set start time
            start_time = tic;
            
            % Pick a sample of size batchsize
            start_index = (inneriter - 1)* batchsize + 1;
            end_index = batchsize*(min(inneriter, options.maxinneriter));
            idx_batchsize = perm_idx(start_index : end_index);
            
            
            % Compute the gradient on this batch.
            partialgrad = getPartialGradient(problem, x, idx_batchsize, storedb, key);
            
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
            
            
            % Update partialgrad
            if strcmp(options.update_type, 'svrg')
                
                % Caclculate transported full batch gradient from x0 to x.
                grad0_transported = problem.M.transp(x0, x, grad0); % Vector transport.
                
                % Caclculate partialgrad at x0
                partialgrad0 = getPartialGradient(problem, x0, idx_batchsize, storedb, key);
                
                % Caclculate transported partialgrad from x0 to x
                partialgrad0_transported = problem.M.transp(x0, x, partialgrad0); % Vector transport.
                
                
                % Update partialgrad to reduce variance by
                % taking a linear combination with old gradients.
                % We make the combination
                % partialgrad + grad0 - partialgrad0.
                partialgrad = problem.M.lincomb(x, 1, grad0_transported, 1, partialgrad);
                partialgrad = problem.M.lincomb(x, 1, partialgrad, -1, partialgrad0_transported);
                
                
            elseif strcmp(options.update_type, 'sg')
                % Do nothing
                
            else
                error(['Unknown options.update_type. ' ...
                    'Should be svrg or sg.']);
                
            end
            
            % Update x
            xnew =  problem.M.retr(x, partialgrad, -stepsize);
            newkey = storedb.getNewKey();
            
            % Elapsed time
            elapsed_time = elapsed_time + toc(start_time);
            
            iter = iter + 1; % Total number updates.
            
            if strcmp(options.update_type, 'svrg') && options.svrg_type == 2 && inneriter == update_instance
                xsave = xnew;
                gradsave = getGrad(problem, xnew);
            end
            
            if options.saveinnerstats
                if canGetCost(problem)
                    newcost = problem.cost(xnew);
                    cost = newcost;
                end
                newpartialgradnorm = problem.M.norm(xnew, partialgrad);
                key = newkey;
                partialgradnorm = newpartialgradnorm;
                innerstats = saveinnerstats();
                innerinfo(inneriter) = innerstats;
                if options.verbosity > 1
                    if canGetCost(problem)
                        fprintf('%s:    %5d (%7d)\t%+.16e\t%.8e\t%.8e\n', options.update_type, epoch, inneriter, cost, partialgradnorm, stepsize);
                    else
                        fprintf('%s:    %5d (%7d)\t%.8e\t%.8e\n', options.update_type, epoch, inneriter, partialgradnorm, stepsize);
                        
                    end
                end
            end
            
            x = xnew;
            key = newkey;
        end
        
        % Calculate cost, grad, and gradnorm
        if strcmp(options.update_type, 'svrg') && options.svrg_type == 2
            x0 = xsave;
            grad0 = gradsave;
        else
            if strcmp(options.update_type, 'svrg')
                tsvrg = tic;
            end
            if strcmp(options.update_type, 'svrg')
                elapsed_time = elapsed_time + toc(tsvrg);
            end
            x0 = xnew;
            grad0 = getGradient(problem, xnew);
        end
        
        % Calculate cost, grad, and gradnorm
        if canGetCost(problem)
            [newcost, newgrad] = getCostGrad(problem, xnew, storedb, newkey);
            cost = newcost;
            newgradnorm = problem.M.norm(xnew, newgrad);
            gradnorm = newgradnorm;
        elseif canGetGradient(problem)
            newgrad = getGradient(problem, xnew, storedb, newkey);
            newgradnorm = problem.M.norm(xnew, newgrad);
            gradnorm = newgradnorm;
        end
        
        % Transfer iterate info
        x = xnew;
        key = newkey;
        
        % Log statistics for freshly executed iteration
        stats = savestats();
        
        if options.saveinnerstats
            stats.innerinfo = innerinfo;
        end
        info(epoch+1)= stats;
        if options.saveinnerstats
            info(epoch+1).innerinfo = innerinfo;
        end
        
        % Reset if boosting used already.
        if toggle == 1
            options.update_type = 'svrg';
        end
        
        % Print output
        if options.verbosity > 0
            if canGetCost(problem) && canGetGradient(problem)
                fprintf('%s:    %5d\t%+.16e\t%.8e\t%.8e\n',options.update_type, epoch, cost, gradnorm, stepsize);
            end
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
        if canGetCost(problem)
            stats.cost = cost;
        end
        if canGetGradient(problem)
            stats.gradnorm = gradnorm;
        end
        if epoch == 0
            stats.time = 0;
        else
            stats.time = info(epoch).time + elapsed_time;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    
    % Save the stats per iteration.
    function innerstats = saveinnerstats()
        innerstats.inneriter = inneriter;
        if inneriter == 0
            if canGetCost(problem)
                innerstats.cost = NaN;
            end
            innerstats.partialgradnorm = NaN;
            innerstats.time = 0;
        else
            if canGetCost(problem)
                innerstats.cost = cost;
            end
            innerstats.partialgradnorm = gradnorm;
            if inneriter == 1
                innerstats.time = elapsed_time;
            else
                innerstats.time = innerinfo(inneriter-1).time + elapsed_time;
            end
        end
        
    end
    
    
end


