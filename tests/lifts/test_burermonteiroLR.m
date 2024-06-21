function test_burermonteiroLR()

    %% Create a problem instance

    % We optimize over matrices of size m-by-n with rank <= r
    m = 1000;
    n = 1000;
    r = 11;

    % The structure Rmn provides functions to operate on large matrices.
    Rmn = euclideanlargefactory(m, n);
    
    % Select some of the m*n entries uniformly at random.
    % The dimension of the set of matrices of size m-by-n with rank r is
    % r(m+n-r). We aim to sample a multiple of that. This multiplier is
    % the oversampling factor (osf).
    desired_osf = 5;
    [I, J, M, actual_osf] = random_mask(m, n, r, desired_osf);
    
    % Generate a ground-truth matrix of rank rtrue (potentially different
    % from r) and compute the selected entries of that matrix, in atrue.
    rtrue = 10;
    Astar.U = stiefelfactory(m, rtrue).rand();
    Astar.V = stiefelfactory(n, rtrue).rand();
    Astar.S = diag(.5 + .5*rand(rtrue, 1));   % singular values of target
    % Astar.S = diag(0.9.^(0:9));
    atrue = Rmn.sparseentries(M, Astar);

    %% Create a problem structure defining the problem in R^(mxn)

    % The problem structure 'downstairs' defines the problem over all mxn
    % matrices, without rank constraint and with efficient use of sparsity.
    downstairs.M = Rmn;
    downstairs.cost = @cost;
    downstairs.grad = @grad;
    downstairs.hess = @hess;
    function store = prepare(X, store)
        if ~isfield(store, 'residue')
            store.residue = Rmn.sparseentries(M, X) - atrue;
            store = incrementcounter(store, 'sparseentries');
        end
    end
    function [f, store] = cost(X, store)
        store = prepare(X, store);
        f = .5*norm(store.residue)^2;
    end
    function [g, store] = grad(X, store)
        store = prepare(X, store);
        % g = sparse(I, J, store.residue);
        g = replacesparseentries(M, store.residue);
    end
    function [h, store] = hess(X, Xdot, store) %#ok<INUSD>
        MXdot = Rmn.sparseentries(M, Xdot);
        % Increment by 2 because Xdot can have rank up to 2r
        store = incrementcounter(store, 'sparseentries', 2);
        % h = sparse(I, J, MXdot);
        h = replacesparseentries(M, MXdot);
    end
    
    % checkgradient(downstairs);
    % checkhessian(downstairs);

    %% Define options for the manopt solvers
    options = struct();

    stats = statscounters({'sparseentries'});
    options.statsfun = statsfunhelper(stats);

    options.theta = sqrt(2)-1;
    options.maxtime = 30;
    options.tolgradnorm = 0;
    options.tolcost = 1e-12;
    
    %% Build an initial guess
    X0_USV.U = stiefelfactory(m, r).rand();
    X0_USV.V = stiefelfactory(n, r).rand();
    X0_USV.S = diag(rand(r, 1))/1000;

    X0_LR = Rmn.to_LR(X0_USV);

    
    %% Lift the downtairs problem through the LR' parameterization
    lift = burermonteiroLRlift(m, n, r);
    upstairs = manoptlift(downstairs, lift);
    [X_LR, ~, LR_info] = trustregions(upstairs, X0_LR, options); %#ok<ASGLU>
    

    %% Lift the downstairs problem through the desingularization
    desing.M = desingularizationfactory(m, n, r);
    desing.cost = @cost;
    desing.egrad = @grad;
    desing.ehess = @ehess_xp;
    function [h, store] = ehess_xp(X, Xdot, store)
        [h, store] = hess(X, desing.M.tangent2ambient(X, Xdot).X, store);
    end

    [X_XP, ~, XP_info] = trustregions(desing, X0_USV, options); %#ok<ASGLU>
    
    
    %% Restrict the downstairs problem to matrices of rank exactly r
    
    fixedrk.M = fixedrankembeddedfactory(m, n, r);
    fixedrk.cost = @cost;
    fixedrk.egrad = @grad;
    fixedrk.ehess = @ehess_fr;
    function [h, store] = ehess_fr(X, Xdot, store)
        [h, store] = hess(X, fixedrk.M.tangent2ambient(X, Xdot), store);
    end
    
    [X_FR, ~, FR_info] = trustregions(fixedrk, X0_USV, options); %#ok<ASGLU>


    %% Plot some statistics to compare the various approaches
    hplt = ...
        semilogy([LR_info.time], [LR_info.cost], '.-', ...
                 [XP_info.time], [XP_info.cost], '.-', ...
                 [FR_info.time], [FR_info.cost], '.-');
    set(hplt, 'LineWidth', 2);
    set(hplt, 'MarkerSize', 20);
    legend('LR', 'desingularisation', 'fixed rank', 'Location', 'northeast');
    xlabel('Time [s]');
    ylabel('Cost function value (training loss)');
    title(sprintf('Oversampling factor: %.4g, Observed fraction: %.4g', ...
                   actual_osf, nnz(M)/numel(M)));
    grid on;

end


% This function aims to select osf*r*(m+n-r) entries uniformly at random
% out of a matrix of size m-by-n, where osf is the oversampling factor.
% The output is a sparse matrix M of size m-by-n with 
function [I, J, M, osf] = random_mask(m, n, r, osf)
    % Expected number of collisions in a random sample of b numbers among a
    % numbers with replacement.
    expected_redundant = @(a, b) ceil(b + a*(((a-1)/a)^b - 1));
    % We'll select a few more to make up for collisions.
    desired_sample_size = round(osf*r*(m+n-r));
    initial_sample_size = desired_sample_size + ...
                            4*expected_redundant(m*n, desired_sample_size);
    sample = unique(randi(m*n, initial_sample_size, 1));
    % With high probability, we have too many in our sample.
    % Trim uniformly at random until we have just the right number.
    while numel(sample) > desired_sample_size
        to_delete = unique(randi(numel(sample), ...
                           numel(sample) - desired_sample_size, 1));
        sample(to_delete) = [];
    end
    % Update the actual sample size and oversampling factor.
    sample_size = length(sample);
    osf = sample_size/(r*(m+n-r));
    % Convert the sampled linear indices into (i, j) pairs and a mask M.
    [I, J] = ind2sub([m, n], sample);
    M = sparse(I, J, ones(sample_size, 1));
end
