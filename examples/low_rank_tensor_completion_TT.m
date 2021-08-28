function low_rank_tensor_completion_TT()
% Example file for the manifold encoded in fixedTTrankfactory.
%
% The script runs a tensor completion task, where tensors are controlled
% by a low Tensor-Train rank. The factory fixedTTrankfactory rests heavily
% on TTeMPS 1.1 (slightly modified for Manopt), coded by M. Steinlechner.
% See manopt/manifolds/ttfixedrank/TTeMPS_1.1/ for license and installation
% instruction (in particular, certain MEX files may require compiling).
%
% This script generates results from figure 1 of the following paper:
%   Michael Psenka and Nicolas Boumal
%   Second-order optimization for tensors with fixed tensor-train rank
%   NeurIPS OPT2020 workshop
%   https://arxiv.org/abs/2011.13395
%
% See also: fixedTTrankfactory

% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Michael Psenka, Jan. 6, 2021
% Contributors: Nicolas Boumal
%
% Change log:

% Set the random seed for reproducible results.
% rng(15);

% order of the tensors
d = 9;

% size vector of the tensors
nn = 4;
n = nn * ones(1, d);

% set of ranks_ to test on. Each rank vector is of the form (1, r, ..., r, 1)
ranks_ = 3;
% set of omega to tensor size ratios we want to observe
omegaRatio_ = 0.1;

prob_dist = [1 1 1 1];

% How many times should the experiments be run? Set to 10 in the paper.
count = 1;

% create a cell for storing final output tensors for each method.
% Want number dependent on whether we vary the ranks or the omega ratios
finalTR_ = cell(1, count);

finalRTTC_ = cell(1, count);
finalALS_ = cell(1, count);

% cell for storing all targets (to test convergence at the end)
% also tracking all omegas and gammas

targets_ = cell(1, count);
omegas_ = cell(1, count);
gammas_ = cell(1, count);

% We want different max inner iter for TR depending on how hard the problem is
maxInner_ = 10000;

% Specify max iter for RTTC at each test
maxIterRTTC_ = 5000;
maxIterALS_ = 1000;

% set to true if you want to verify condition numbers
computeCondition = false;
% variable to store target Hessian condition numbers
cond_nums = [];

for p = 1:count

    rr = ranks_;

    r = [1, rr * ones(1, d - 1), 1];
    r(3:8) = ones(1, 6) * 5;
    r(4:7) = ones(1, 4) * 10;
    r(5:6) = ones(1, 2) * 10;

    rTarg = r;

    % options for Steinlechner's algorithms, ALS and Riemannian respectively
    opts = struct('maxiter', maxIterALS_, 'tol', 1e-14, 'reltol', 0, 'gradtol', 0);
    opts_tt = struct('maxiter', maxIterRTTC_, 'tol', 1e-14, 'reltol', 0, 'gradtol', 1e-8);

    % set of observed points for tensor completion (Omega) and test set (Gamma)
    % test set used to make sure algorithms converge to the right tensor
    sizeOmega = round(omegaRatio_ * prod(n));

    sizeGamma = sizeOmega;

    Omega = makeOmegaSet_local(n, sizeOmega, prob_dist);
    Gamma = makeOmegaSet_local(n, sizeGamma);

    omegas_{p} = Omega;
    gammas_{p} = Gamma;

    A = TTeMPS_randn(rTarg, n);
    targets_{p} = A;

    % vector representing observed points of A at Omega and Gamma
    A_Omega = A(Omega);
    A_Gamma = A(Gamma);

    % Starting point for optimization. Forced to be unit norm
    X0 = TTeMPS_randn(r, n);
    X0 = (1 / norm(X0)) * X0;
    X0 = orthogonalize(X0, X0.order);

    % Construction of ManOpt factory for fixed-rank TT manifold
    % n is the dimension vector, r the rank vector, and Omega an optional
    % parameter to specify which points we observe on the manifold
    TT = fixedTTrankfactory(n, r, Omega);

    % checkmanifold(TT)

    disp("Oversampling ratio: " + sizeOmega / TT.dim());

    % Set up two problems: the normal tensor completion problem (problem)
    % and the same problem with L2 regularization (rProblem)
    rProblem.M = TT;
    problem.M = TT;

    % Setting up the original tensor competion problem for ManOpt
    problem.cost = @(x) eCostCompl(x, A_Omega, Omega);
    problem.egrad = @(x) eGradCompl(x, A_Omega, Omega);
    problem.ehess = @(x, u) eHessCompl(u, Omega);

    % computes the spectrum of the Hessian at current target point
    if computeCondition
        A_base = orthogonalize(A, A.order);
        spec = hessianspectrum(problem, A_base);
        cond_nums(end + 1) = spec(end) / spec(1);
    end

    problem = rmfield(problem, 'ehess');

    % options for trust regions
    options.Delta0 = 100;
    options.Delta_bar = 100 * 2^11;
    options.maxiter = 250;
    options.maxinner = maxInner_;
    options.maxtime = inf;
    options.tolgradnorm = 1e-8;

    % setting up stats func for test cost
    problem.Gamma = Gamma;
    problem.A_Gamma = A_Gamma;
    options.statsfun = @test_cost_manopt;


    % Solve tensor completion problem w/ finite differences
    [finalTR, cost_man_fd{p}, stats_man_fd{p}] = trustregions(problem, X0, options);

    problem.ehess = @(x, u) eHessCompl(u, Omega);
    % now solve with analytic hessian
    [finalTR, cost_man{p}, stats_man{p}] = trustregions(problem, X0, options);

    % Final RTTC, slightly change parameters to allow lower gradient tolerance

    [finalRTTC_{p}, cost_tt{p}, test_tt{p}, stats_tt{p}] = ...
        completion_orth(A_Omega, Omega, A_Gamma, Gamma, X0, opts_tt);

    % ALS completion
    [finalALS_{p}, cost_als{p}, test_als{p}, stats_als{p}] = ...
        completion_als(A_Omega, Omega, A_Gamma, Gamma, X0, opts);

end

%%
l = lines(7);
midred = l(end, :);
darkred = brighten(l(end, :), -0.7);
lightred = brighten(midred, 0.7);

midblue = l(1, :);
darkblue = brighten(midblue, -0.7);
lightblue = brighten(midblue, 0.7);

% set alpha
lightred(end + 1) = 0.7;
lightblue(end + 1) = 0.7;
midred(end + 1) = 0.7;
midblue(end + 1) = 0.7;

figure;

for k = 1:count
    A = targets_{k};
    Omega = gammas_{k};
    A_Omega = A(Omega);
    semilogy([stats_man_fd{k}.time], sqrt(2 * [stats_man_fd{k}.cost_test]) / norm(A_Omega), 'color', midred, 'linewidth', 2)
    hold on
    semilogy([stats_man{k}.time], sqrt(2 * [stats_man{k}.cost_test]) / norm(A_Omega), 'color', midblue, 'linewidth', 2)
    semilogy(stats_tt{k}.time, test_tt{k}, 'color', lightred, 'linewidth', 2)
    semilogy(stats_als{k}.time, test_als{k}, 'color', lightblue, 'linewidth', 2)
end

legend({'FD-TR', 'RTR', 'RTTC', 'ALS'})
xlabel('Time (s)')
ylabel('Test Cost')

figure;

for k = 1:count
    A = targets_{k};
    Omega = omegas_{k};
    A_Omega = A(Omega);
    semilogy([stats_man_fd{k}.time], sqrt(2 * [stats_man_fd{k}.cost]) / norm(A_Omega), 'color', midred, 'linewidth', 2)
    hold on
    semilogy([stats_man{k}.time], sqrt(2 * [stats_man{k}.cost]) / norm(A_Omega), 'color', midblue, 'linewidth', 2)
    semilogy(stats_tt{k}.time, cost_tt{k}, 'color', lightred, 'linewidth', 2)
    semilogy(stats_als{k}.time, cost_als{k}, 'color', lightblue, 'linewidth', 2)
end

legend({'FD-TR', 'RTR', 'RTTC', 'ALS'})
xlabel('Time (s)')
ylabel('Training Cost')


figure;

for k = 1:count
    A = targets_{k};
    Omega = omegas_{k};
    A_Omega = A(Omega);
    semilogy([stats_man_fd{k}.time], [stats_man_fd{k}.gradnorm] / stats_tt{k}.gradnorm(1), 'color', midred, 'linewidth', 2)
    hold on
    semilogy([stats_man{k}.time], [stats_man{k}.gradnorm] / stats_tt{k}.gradnorm(1), 'color', midblue, 'linewidth', 2)
    semilogy(stats_tt{k}.time, (stats_tt{k}.gradnorm) / stats_tt{k}.gradnorm(1), 'color', lightred, 'linewidth', 2)
end

legend({'FD-TR', 'RTR', 'RTTC'})
xlabel('Time (s)')
ylabel('Gradient Norm')


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Stats function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function stats = test_cost_manopt(problem, x, stats)
    stats.cost_test = .5 * norm(x(problem.Gamma) - problem.A_Gamma)^2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS FOR MANOPT TRUST REGIONS %%%%%%%%%%%%%%%%%%%%%%%%%

% Non-regularized Euclidean functions
function c = eCostCompl(x, A, A0)
    c = .5 * norm(x(A0) - A)^2;
end

function g = eGradCompl(x, A, A0)
    g = (x(A0) - A);
end

function h = eHessCompl(u, A0)
    uTT = tangent_to_TTeMPS(u);
    h = uTT(A0);
end

%%%%%%%%%%%%%%%% CUSTOM OMEGA SET GENERATOR, NON_UNIFORM DIST. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dist is given distribution for the integers. If none given, default to uniform
function Omega = makeOmegaSet_local(n, sizeOmega, dist)

    if sizeOmega > prod(n)
        error('makeOmegaSet:sizeOmegaTooHigh', 'Requested size of Omega is bigger than the tensor itself!')
    end

    d = length(n);
    subs = zeros(sizeOmega, d);

    for i = 1:d

        if nargin == 2
            subs(:, i) = randi(n(i), sizeOmega, 1);
        else
            subs(:, i) = randsample(n(i), sizeOmega, true, dist);
            % for k = 1:sizeOmega
            %     subs(k,i) = randsample(n(i), sizeOmega, true, dist);

        end

    end

    Omega = unique(subs, 'rows');
    m = size(Omega, 1);

    while m < sizeOmega
        subs(1:m, :) = Omega;

        for i = 1:d

            if nargin == 2
                subs(m + 1:sizeOmega, i) = randi(n(i), sizeOmega - m, 1);
            else
                subs(m + 1:sizeOmega, i) = randsample(n(i), sizeOmega - m, true, dist);
            end

        end

        Omega = unique(subs, 'rows');
        m = size(Omega, 1);
    end

end
