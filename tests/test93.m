function test93()

% Test for fixedrankGHquotientfactory geometry (low rank completion)
%
% CG comparison for Test 18

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 

clc; close all;
%     reset(RandStream.getDefaultStream);
%     randnfoo = randn(1, 1); %#ok<NASGU>

m = 800;
n = 800;
r = 5;
A = randn(m, r);
B = randn(n, r);
C = A*B';

problem.M = fixedrankGHquotientfactory(m, n, r);

df = problem.M.dim();
p = 8*df/(m*n);
mask = rand(m, n) <= p;

symm = @(M) .5*(M+M');

problem.cost = @cost;
    function f = cost(X)
        f = .5*norm(mask.*(X.G*X.H' - C), 'fro')^2;
    end

problem.grad = @grad;
    function g = grad(X)
        G = X.G;
        H = X.H;
        GtG = X.G'*X.G;
        HtH = X.H'*X.H;
        
        R = mask.^2 .* (G*H' - C);
        g = struct('G', R*H*(GtG), ...
            'H', R'*G*(HtH) );
    end

problem.hess = @hess;
    function Hess = hess(X, eta)
        
        S = mask.*( X.G*X.H' - C);
        S_star  = mask.*(eta.G*X.H' + X.G*eta.H');
        GtG = X.G'*X.G;
        HtH = X.H'*X.H;
        
        Hess.G = S_star*X.H*(GtG);
        Hess.H = S_star'*X.G*(HtH);
        
        ShH = S*eta.H;
        ShG = S'*eta.G;
        SH = S*X.H;
        SG = S'*X.G;
        
        Hess.G = Hess.G + 2*SH*symm(eta.G'*X.G) + ShH*(GtG);
        Hess.H = Hess.H + 2*SG*symm(eta.H'*X.H) + ShG*(HtH);
        
        gradG = SH*(GtG);
        gradH = SG*(HtH);
        GgradG = X.G'*gradG;
        HgradH = X.H'*gradH;
        
        Hess.G = Hess.G - gradG*((GtG)\symm(X.G'*eta.G)) - eta.G*(GtG \ symm(GgradG)) + X.G*((GtG)\symm(eta.G'*gradG));
        Hess.H = Hess.H - gradH*((HtH)\symm(X.H'*eta.H)) - eta.H*(HtH \ symm(HgradH)) + X.H*((HtH)\symm(eta.H'*gradH));
        
        Hess = problem.M.proj(X, Hess);
        
    end


%     checkgradient(problem);
%     drawnow;
%     pause;
%     checkhessian(problem);
%     drawnow;
%     pause;
%
%     problem = rmfield(problem, 'hess');

% [U S V] = svds(mask.*C, r);
% G0 = U*(S.^0.5);
% H0 = V*(S.^0.5);
G0 = randn(m, r);
H0 = randn(n, r);

X0 = struct('G', G0, 'H', H0 );
options.statsfun = @statsfun;

    function stats = statsfun(problem, x, stats)
        stats.RMSE = nan;%norm(C - x.M*x.N', 'fro')/sqrt(m*n);
    end

options.maxiter = inf;
options.maxinner = 30;
options.maxtime = 120;
%     options.mininner = problem.M.dim();
options.tolgradnorm = 1e-5;
options.Delta_bar = 10*r;
options.Delta0 = r;
%     options.useRand = true;
%     [Xopt costopt info] = trustregions(problem, X0, options);
%      [Xopt costopt info] = steepestdescent(problem, X0, options);

fprintf('----------- Steepest descent with NW -----------\n');
options.linesearch = @linesearch;
[Xopt costopt infos_SD] = steepestdescent(problem, X0, options);


fprintf('----------- Steepest descent with adaptive -----------\n');
options.linesearch = @linesearch_adaptive;
[Xopt costopt infos_SD_adaptive] = steepestdescent(problem, X0, options);

options.beta_type = 'P-R'; % Other options are 'steep' and 'FR'
options.orth_value = Inf;


fprintf('----------- CG algorithm with NW -----------\n');
options.linesearch = @linesearch;
[Xopt costopt infos_CG] = conjugategradient(problem, X0, options);


fprintf('----------- CG algorithm with adaptive -----------\n');
options.linesearch = @linesearch_adaptive;
[Xopt costopt infos_CG_adaptive] = conjugategradient(problem, X0, options);



%     keyboard;

% subplot(3, 1, 1);
% semilogy([info.iter], [info.cost], '.-');
% subplot(3, 1, 2);
% semilogy([info.iter], [info.gradnorm], '.-');
% subplot(3, 1, 3);
% semilogy([info.iter], [info.RMSE], '.-');



% Cost versus iterations
fs = 20;
figure;
semilogy(0:length([infos_SD.cost])-1,[infos_SD.cost],'Color','b','LineWidth',2);
hold on;
semilogy(0:length([infos_SD_adaptive.cost])-1,[infos_SD_adaptive.cost],'Color','r','LineWidth',2);
semilogy(0:length([infos_CG.cost])-1,[infos_CG.cost],'--','Color','b','LineWidth',2);
semilogy(0:length([infos_CG_adaptive.cost])-1,[infos_CG_adaptive.cost],'--','Color','r','LineWidth',2);

hold off;
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'Number of iterations','FontSize',fs);
ylabel(ax1,'Cost ','FontSize',fs);
mincost = min([infos_SD.cost]);
maxcost = max([infos_SD.cost]);
axis([get(gca,'XLim') mincost maxcost])
legend('SD NW','SD adaptive', 'CG NW','CG adaptive');
legend 'boxoff';
box off;




end