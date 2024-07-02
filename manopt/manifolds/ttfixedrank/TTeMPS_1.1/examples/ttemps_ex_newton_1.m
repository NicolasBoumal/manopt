% Example for the algorithms described in 
%
%       D. Kressner, M. Steinlechner, A. Uschmajew:
%		Low-rank tensor methods with subspace correction for symmetric eigenvalue problems
%		SIAM J. Sci. Comput., 36(5):A2346-A2368, 2014.
%
% Code to produce Figure 4.1, the smallest eigenvalue of the discretized Newton potential
% (without the no-preconditioner graph, as it takes very long to compute and should not be used anyway)
% =========================================================================

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

clear all
close all

n = 128;
d = 10;
A = newton_potential( n, d );
p = 1;
r = 1;
maxrank = 40;

% Run standard ALS procedure:
% =========================================================================

if ~exist('newt_1_blk.mat','file')
    opts = struct( 'maxiter', 3, ...
                   'maxrank', maxrank, ...
                   'tol', 1e-8, ... 
                   'tolOP', 1e-3, ... 
                   'tolLOBPCG', 1e-6, ... 
                   'maxiterLOBPCG', 2000, ... 
                   'verbose', 1 , ...
                   'precInner', true);

    rng(11)
    rr = [1, 8 * ones(1, d-1), 1];
    [X_blk, C_blk, evalue_blk, residuums_blk, micro_res_blk, objective_blk, t_blk] = block_eigenvalue( A, p, rr, opts);
    save('newt_1_blk', 'X_blk', 'C_blk', 'evalue_blk', 'residuums_blk', 'micro_res_blk', 'objective_blk','t_blk');
else
    load('newt_1_blk.mat')
end

% Run EVAMEn:
% =========================================================================

if ~exist('newt_1_evamen.mat','file')
    opts = struct( 'maxiter', 3, ...
                   'maxrank', maxrank, ...
                   'maxrankRes', 0, ...
                   'tol', 1e-8, ... 
                   'tolOP', 1e-3, ... 
                   'tolLOBPCG', 1e-6, ... 
                   'maxiterLOBPCG', 2000, ... 
                   'verbose', 1 , ...
                   'precInner', true);
    rng(11)
    rr = [1, 1 * ones(1, d-1), 1];
    [X_evamen, C_evamen, evalue_evamen, residuums_evamen, micro_res_evamen, objective_evamen, t_evamen] = amen_eigenvalue( A, 1, p, rr, opts);
    save('newt_1_evamen', 'X_evamen', 'C_evamen', 'evalue_evamen', 'residuums_evamen', 'micro_res_evamen', 'objective_evamen','t_evamen');
else
    load('newt_1_evamen.mat')
end


% Prepare data for plotting:
% =========================================================================

evalue_end = repmat(evalue_blk(:,end), [1,size(evalue_blk,2)-1]);
evalue_end = evalue_blk(end);
ev_blk = abs(evalue_blk(:,1:end-1) - evalue_end);
ev_evamen = abs(evalue_evamen(:,1:end-1) - evalue_end);


% Plot vs. Iterations
% =========================================================================

f = figure
set(0,'defaultlinelinewidth',2)
subplot(1,2,1)
semilogy( sqrt(sum(micro_res_blk.^2, 1)), '-b' )
hold on
semilogy( sqrt(sum(micro_res_evamen.^2, 1)), '-k' )
%
semilogy( sum(ev_blk,1), '--b' )
semilogy( sum(ev_evamen,1), '--k' )

res_blk         = sqrt(sum(micro_res_blk.^2, 1))
res_evamen  = sqrt(sum(micro_res_evamen.^2, 1))
%
semilogy((d-1):(d-1):length(micro_res_blk),res_blk(:,(d-1):(d-1):end),'ob')
semilogy((d-1):(d-1):length(micro_res_evamen),res_evamen(:,(d-1):(d-1):end),'ok')
semilogy((d-1):(d-1):length(ev_blk),sum(ev_blk(:,(d-1):(d-1):end),1),'ob')
semilogy((d-1):(d-1):length(ev_evamen),sum(ev_evamen(:,(d-1):(d-1):end),1),'ok')

h_leg = legend('Res. err., Block-ALS',... 
       'Res. err. EVAMEn, local prec.',...
       'EV. err., Block-ALS',... 
       'EV. err. EVAMEn, local prec.')

set(gca,'fontsize',16)
set(h_leg, 'fontsize',12)
xlabel('Microiterations')
ylabel('Residual and eigenvalue error')  


% Plot vs. Time
% =========================================================================

subplot(1,2,2)
semilogy( t_blk, sqrt(sum(micro_res_blk.^2, 1)), '-b' )
hold on
semilogy( t_evamen, sqrt(sum(micro_res_evamen.^2, 1)), '-k' )

semilogy( t_blk, sum(ev_blk,1), '--b' )
semilogy( t_evamen, sum(ev_evamen,1), '--k' )

semilogy(t_blk((d-1):(d-1):end),res_blk(:,(d-1):(d-1):end),'ob')
semilogy(t_evamen((d-1):(d-1):end),res_evamen(:,(d-1):(d-1):end),'ok')

semilogy(t_blk((d-1):(d-1):end),sum(ev_blk(:,(d-1):(d-1):end),1),'ob')
semilogy(t_evamen((d-1):(d-1):end),sum(ev_evamen(:,(d-1):(d-1):end),1),'ok')

semilogy(t_blk((d-1):(d-1):end),        res_blk(:,(d-1):(d-1):end),'ob')
semilogy(t_evamen((d-1):(d-1):end), res_evamen(:,(d-1):(d-1):end),'ok')

semilogy(t_blk((d-1):(d-1):end),        sum(ev_blk(:,(d-1):(d-1):end),1),'ob')
semilogy(t_evamen((d-1):(d-1):end), sum(ev_evamen(:,(d-1):(d-1):end),1),'ok')

xlim([0,t_blk(end)])
h_leg = legend('Res. err., Block-ALS',... 
       'Res. err. EVAMEn, local prec.',...
       'EV. err., Block-ALS',... 
       'EV. err. EVAMEn, local prec.')
set(gca,'fontsize',16)
set(h_leg, 'fontsize',12)
xlabel('Time [s]')
ylabel('Residual and eigenvalue error')  


set(f, 'Position', [0 0 1200 700])
