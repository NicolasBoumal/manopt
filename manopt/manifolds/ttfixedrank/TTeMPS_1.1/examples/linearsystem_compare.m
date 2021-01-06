% Example code for the algorithms described in 
%
%   D. Kressner, M. Steinlechner, and B. Vandereycken.
%   Preconditioned low-rank Riemannian optimization for linear systems with tensor product structure.
%   Technical report, July 2015. Revised February 2016. To appear in SIAM J. Sci. Comput.

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

clear all;
close all;

rng(11);
n = 20;
d = 5;
r = 10;

nn = n*ones(1,d);
%rr = [1, r*ones(1,d-1), 1];
rr = [1 15 20 20 15 1];

% Choose one.
%operatorname = 'diffusion';
operatorname = 'newton';

if strcmpi(operatorname,'diffusion')
    A = anisotropicdiffusion( n, d );
elseif strcmpi(operatorname,'newton')
    A = newton_potential( n, d );
else
    error('Unknown operator requested.')
end

filename = [operatorname, '_d', num2str(d), '_n', num2str(n), '_r', num2str(r)];


if ~exist(['amen/', filename, '.mat'],'file')

    rng(11);

    % Lapl_op   : pure dD Laplacian as TTeMPS_op_laplace for preconditioning
    Riem_CG_TOL = 1e-5; % can be low
    ALS_CG_TOL = 1e-10; % cannot be too low, stagnates for d = 10; n = 100; r = 40; otherwise
    
    Lapl_op = TTeMPS_op_laplace( A.L0, d ); 
    Lapl_op = initialize_precond( Lapl_op );
  
    % rhs is rank one of all ones, scaled to Frobenius norm one
    % setup rhs as rank 1 of all ones
    rr1 = [1, 1*ones(1,d-1), 1];
    F = TTeMPS_rand( rr1, nn );
    for i = 1:d
        F.U{i} = ones(size(F.U{i}));
    end
    F = 1/norm(F) * F;

    % Choose operators: 
    %    A is actual system
    %    prec_L_op is the approximate Hessian that will be solved
    %    prec_P_op is the preconditioner for prec_L_op. HAS TO BE of Laplace structure
    prec_L_op = Lapl_op;
    prec_P_op = Lapl_op;

    % One micro step of ALS decreases the error by a huge
    % factor. Make the result of one micro step the initial guess.
    X0 = construct_initial_guess(A, F, rr, nn);
    X0_amen = construct_initial_guess(A, F, [1,1*ones(1,d-1),1], nn);

    % ==============================
    % TEST CASE 1: SIMPLE AMEn
    % ==============================
    disp('TEST CASE 1: Prec. AMEn, max rank res = 4');
    % !!! ALS is always preconditioned with Laplacian part
    opts = struct( 'nSweeps', 3, ...
                   'solver', 'pcg', ...
                   'maxrankRes', 4, ...
                   'prec', true)
                   
    tic
    [X_amen_prec1, residuum_amen_prec1, cost_amen_prec1, times_amen_prec1] = amen_fast( A, F, X0_amen, opts )
    t_amen_prec1 = toc;

    % ==============================
    % TEST CASE 1: SIMPLE AMEn
    % ==============================
    disp('TEST CASE 1: Prec. AMEn, max rank res = 8');
    % !!! ALS is always preconditioned with Laplacian part
    opts = struct( 'nSweeps', 3, ...
                   'solver', 'pcg', ...
                   'maxrankRes', 8, ...
                   'prec', true)
                   
    tic
    [X_amen_prec2, residuum_amen_prec2, cost_amen_prec2, times_amen_prec2] = amen_fast( A, F, X0_amen, opts )
    t_amen_prec2 = toc;
    
    
    % ==============================
    % TEST CASE 1: SIMPLE ALS
    % ==============================
    disp('TEST CASE 1: Simple ALS');
    % !!! ALS is always preconditioned with Laplacian part
    opts = struct( 'nSweeps', 3, ...
                   'solver', 'pcg', ...
                   'pcg_accuracy', ALS_CG_TOL);
                   
    tic
    [X_als, residuum_als, cost_als, times_als] = alsLinsolve_fast( A, F, X0, opts )
    t_als = toc;

    % ==============================
    % TEST CASE 2: Riemannian SD with 1 application of Prec.
    % ==============================
    disp('TEST CASE 2: Riemannian SD with 1 steps of the approx. prec.');
    opts = struct('maxiter', 30, ...
                  'precond_tol', Riem_CG_TOL, ...
                  'precond_maxit', 1)
    
    opts.precond_tol = Riem_CG_TOL;
    opts.precond_maxit = 1; % 1 application of prec_P_op
        
    tic;
    [X_SD1, residuum_SD1, gradnorm_SD1, cost_SD1, times_SD1] = RiemannLinsolve( A, F, X0, prec_L_op, prec_P_op, opts )
    t_pcg = toc;
    
    % ==============================
    %% TEST CASE 4: Truncated (Gauss-)Newton with approx. Prec
    %% ==============================
    disp('TEST CASE 4: Truncated (Gauss-)Newton.');
    opts = struct('maxiter', 20, ...
                  'truncatedNewton', true);
    tic
    [X_tn, residuum_tn, gradnorm_tn, cost_tn, times_tn] = RiemannLinsolve( A, F, X0, prec_L_op, prec_P_op, opts )
    t_tn = toc;

    save([filename, '.mat'], 'X_als','residuum_als', 'cost_als', 'times_als', ...
                   'X_amen_prec1','residuum_amen_prec1', 'cost_amen_prec1', 'times_amen_prec1', ...
                   'X_amen_prec2','residuum_amen_prec2', 'cost_amen_prec2', 'times_amen_prec2', ...
                   'X_SD1','residuum_SD1', 'gradnorm_SD1', 'cost_SD1', 'times_SD1', ...
                   'X_tn', 'residuum_tn', 'gradnorm_tn', 'cost_tn', 'times_tn')
else
    load(['amen/', filename, '.mat'])
end


% setup plotting
col = lines(5);
leg = {};

% dummy plot for legend entry
figure(1)
semilogy(1,residuum_als(1), '-o', 'Color', col(1,:),'linewidth',2)
hold on
figure(2)
semilogy(times_als(1), residuum_als(1), '-o', 'Color', col(1,:),'linewidth',2)
hold on
drawnow

figure(1)
semilogy(1,residuum_amen_prec1(1), '-o', 'Color', col(3,:),'linewidth',2)
hold on
figure(2)
semilogy(times_amen_prec1(1), residuum_amen_prec1(1), '-o', 'Color', col(3,:),'linewidth',2)
hold on
drawnow

figure(1)
semilogy(1,residuum_amen_prec2(1), '-o', 'Color', col(2,:),'linewidth',2)
hold on
figure(2)
semilogy(times_amen_prec2(1), residuum_amen_prec2(1), '-o', 'Color', col(2,:),'linewidth',2)
hold on
drawnow

% Start drawing the graphs
leg = [leg, 'ALS', 'AMEn, max. rank res. = 4', 'AMEn, max. rank res. = 8'];

figure(1)
semilogy(residuum_SD1, '--', 'Color', col(5,:),'linewidth',2)
figure(2)
semilogy(times_SD1, residuum_SD1, '--', 'Color', col(5,:),'linewidth',2)
drawnow

leg = [leg, 'Prec. steepest descent'];

figure(1)
semilogy(residuum_tn, '-k','linewidth',2)
figure(2)
semilogy(times_tn, residuum_tn, '-k','linewidth',2)
leg = [leg, 'Approx. Newton'];
drawnow

figure(1)
semilogy(residuum_als, '-', 'Color', col(1,:),'linewidth',2)
iterations = 1:length(residuum_als);
semilogy(iterations(1:d:end),residuum_als(1:d:end), 'o', 'Color', col(1,:),'linewidth',2)
hold on
figure(2)
semilogy(times_als, residuum_als, '-', 'Color', col(1,:),'linewidth',2)
semilogy(times_als(1:d:end), residuum_als(1:d:end), 'o', 'Color', col(1,:),'linewidth',2)
hold on
drawnow

figure(1)
semilogy(residuum_amen_prec1, '-', 'Color', col(3,:),'linewidth',2)
iterations = 1:length(residuum_amen_prec1);
semilogy(iterations(1:d:end),residuum_amen_prec1(1:d:end), 'o', 'Color', col(3,:),'linewidth',2)
hold on
figure(2)
semilogy(times_amen_prec1, residuum_amen_prec1, '-', 'Color', col(3,:),'linewidth',2)
semilogy(times_amen_prec1(1:d:end), residuum_amen_prec1(1:d:end), 'o', 'Color', col(3,:),'linewidth',2)
hold on
drawnow

figure(1)
semilogy(residuum_amen_prec2, '-', 'Color', col(2,:),'linewidth',2)
iterations = 1:length(residuum_amen_prec2);
semilogy(iterations(1:d:end),residuum_amen_prec2(1:d:end), 'o', 'Color', col(2,:),'linewidth',2)
hold on
figure(2)
semilogy(times_amen_prec2, residuum_amen_prec2, '-', 'Color', col(2,:),'linewidth',2)
semilogy(times_amen_prec2(1:d:end), residuum_amen_prec2(1:d:end), 'o', 'Color', col(2,:),'linewidth',2)
hold on
drawnow

figure(1)
legend(leg);
set(gca,'fontsize',16)
xlabel('Iterations')
ylabel('Relative residual')
set(gca,'fontsize',16)
ylim([1e-10,1e1])

figure(2)
legend(leg);
set(gca,'fontsize',16)
xlabel('Time [s]')
ylabel('Relative residual')
set(gca,'fontsize',16)
ylim([1e-10,1e1])
