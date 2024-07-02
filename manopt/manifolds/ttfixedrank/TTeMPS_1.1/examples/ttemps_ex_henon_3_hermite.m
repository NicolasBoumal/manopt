% Example for the algorithms described in 
%
%       D. Kressner, M. Steinlechner, A. Uschmajew:
%		Low-rank tensor methods with subspace correction for symmetric eigenvalue problems
%		SIAM J. Sci. Comput., 36(5):A2346-A2368, 2014.
%
% Code to produce Figure 4.4: Three Eigenvalues for Henon-Heiles with 
% with n=28, d=10

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
% =========================================================================
clear all
close all

n = 28;
d = 10;
A = TTeMPS_op_NN_hermite(n, d);
p = 3;
r = 1;

% Run block eigenvalue procedure:
% =========================================================================

if ~exist('hh_3_blk_hermite.mat','file')
    opts = struct( 'maxiter', 3, ...
                   'maxrank', 40, ...
                   'tol', 1e-8, ... 
                   'tolOP', 1e-3, ... 
                   'tolLOBPCG', 1e-6, ... 
                   'maxiterLOBPCG', 500, ... 
                   'verbose', true , ...
                   'precInner', true);

    rng(11)
    rr = [1, 1 * ones(1, d-1), 1];
    [X_blk_hermite, C_blk_hermite, evalue_blk_hermite, residuums_blk_hermite, micro_res_blk_hermite, objective_blk_hermite, t_blk_hermite] = block_eigenvalue( A, p, rr, opts);
    save('hh_3_blk_hermite', 'X_blk_hermite', 'C_blk_hermite', 'evalue_blk_hermite', 'residuums_blk_hermite', 'micro_res_blk_hermite', 'objective_blk_hermite','t_blk_hermite');
else
    load('hh_3_blk_hermite.mat')
end

% Run EVAMEn:
% =========================================================================

if ~exist('hh_3_evamen_hermite.mat','file')
    opts = struct( 'maxiter', 3, ...
                   'maxrank', 40, ...
                   'maxrankRes', 2, ...
                   'tol', 1e-8, ... 
                   'tolOP', 1e-3, ... 
                   'tolLOBPCG', 1e-6, ... 
                   'maxiterLOBPCG', 500, ... 
                   'verbose', true , ...
                   'precInner', true);
    rng(11)
    rr = [1, 1 * ones(1, d-1), 1];
    [X_evamen_hermite, C_evamen_hermite, evalue_evamen_hermite, residuums_evamen_hermite, micro_res_evamen_hermite, objective_evamen_hermite, t_evamen_hermite] = amen_eigenvalue( A, 1, p, rr, opts);
    save('hh_3_evamen_hermite', 'X_evamen_hermite', 'C_evamen_hermite', 'evalue_evamen_hermite', 'residuums_evamen_hermite', 'micro_res_evamen_hermite', 'objective_evamen_hermite','t_evamen_hermite');
else
    load('hh_3_evamen_hermite.mat')
end


% Prepare data for plotting:
% =========================================================================

evalue_end = repmat(evalue_blk_hermite(:,end), [1,size(evalue_blk_hermite,2)-1]);
ev_blk_hermite = abs(evalue_blk_hermite(:,1:end-1) - evalue_end);
ev_evamen_hermite = abs(evalue_evamen_hermite(:,1:end-1) - evalue_end);


% Plot vs. Iterations
% =========================================================================
f = figure
set(0,'defaultlinelinewidth',2)
subplot(1,2,1)

semilogy( sqrt(sum(micro_res_blk_hermite.^2, 1)), '-b' )
hold on
semilogy( sqrt(sum(micro_res_evamen_hermite.^2, 1)), '-k' )

semilogy( sum(ev_blk_hermite,1), '--b' )
semilogy( sum(ev_evamen_hermite,1), '--k' )

res_blk_hermite         = sqrt(sum(micro_res_blk_hermite.^2, 1))
res_evamen_hermite  = sqrt(sum(micro_res_evamen_hermite.^2, 1))

semilogy((d-1):(d-1):length(micro_res_blk_hermite),res_blk_hermite(:,(d-1):(d-1):end),'ob')
semilogy((d-1):(d-1):length(micro_res_evamen_hermite),res_evamen_hermite(:,(d-1):(d-1):end),'ok')

semilogy((d-1):(d-1):length(ev_blk_hermite),sum(ev_blk_hermite(:,(d-1):(d-1):end),1),'ob')
semilogy((d-1):(d-1):length(ev_evamen_hermite),sum(ev_evamen_hermite(:,(d-1):(d-1):end),1),'ok')

set(gca,'fontsize',20)
xlabel('Microiterations')
ylabel('Residual and eigenvalue error')  

% Plot vs. Time
% =========================================================================

subplot(1,2,2)
semilogy( t_blk_hermite, sqrt(sum(micro_res_blk_hermite.^2, 1)), '-b' )
hold on
semilogy( t_evamen_hermite, sqrt(sum(micro_res_evamen_hermite.^2, 1)), '-k' )

semilogy( t_blk_hermite, sum(ev_blk_hermite,1), '--b' )
semilogy( t_evamen_hermite, sum(ev_evamen_hermite,1), '--k' )

semilogy(t_blk_hermite((d-1):(d-1):end),res_blk_hermite(:,(d-1):(d-1):end),'ob')
semilogy(t_evamen_hermite((d-1):(d-1):end),res_evamen_hermite(:,(d-1):(d-1):end),'ok')

semilogy(t_blk_hermite((d-1):(d-1):end),sum(ev_blk_hermite(:,(d-1):(d-1):end),1),'ob')
semilogy(t_evamen_hermite((d-1):(d-1):end),sum(ev_evamen_hermite(:,(d-1):(d-1):end),1),'ok')

semilogy(t_blk_hermite((d-1):(d-1):end),        res_blk_hermite(:,(d-1):(d-1):end),'ob')
semilogy(t_evamen_hermite((d-1):(d-1):end), res_evamen_hermite(:,(d-1):(d-1):end),'ok')

semilogy(t_blk_hermite((d-1):(d-1):end),        sum(ev_blk_hermite(:,(d-1):(d-1):end),1),'ob')
semilogy(t_evamen_hermite((d-1):(d-1):end), sum(ev_evamen_hermite(:,(d-1):(d-1):end),1),'ok')


h_leg = legend('Res. err., Block-ALS',... 
       'Res. err. EVAMEn, prec.',...
       'EV. err., Block-ALS',... 
       'EV. err. EVAMEn, prec.')
set(gca,'fontsize',20)
set(h_leg, 'fontsize',16)
xlabel('Time [s]')
ylabel('Residual and eigenvalue error')  

set(f, 'Position', [0 0 1200 700])
