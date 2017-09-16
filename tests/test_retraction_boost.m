%% Sept., 16, 2017
%  Testing whether a retraction can be boosted into
%  a higher order retraction by integration (time stepping).
%  It seems yes -- there ought to be more efficient ways.

clear all, close all, clc;

%%
n = 5;
M = spherefactory(n);

x = M.rand();
v = M.randvec(x);

%%

h = logspace(-6, 0, 101);

g = zeros(3, numel(h));

for k = 1 : numel(h)
    
    t = h(k);
    
    g(1, k) = M.dist(M.exp(x, v, t), M.retr(x, v, t));
    
    K = ceil(M.norm(x, v)*t / 1e-4); % 1000; % M.norm(x, u)*t;
    u = v;
    y0 = x;
    y1 = M.retr(x, u, (1/K)*t);
    for KK = 1 : (K-1)
        u = M.isotransp(y0, y1, u);
        y2 = M.retr(y1, u, (1/K)*t);
        y0 = y1;
        y1 = y2;
    end
    g(2, k) = M.dist(M.exp(x, v, t), y1);
    
    K = ceil(M.norm(x, v)*t / 1e-4); % 1000; % M.norm(x, u)*t;
    u = v;
    y0 = x;
    y1 = M.retr(x, u, (1/K)*t);
    for KK = 1 : (K-1)
        u = M.transp(y0, y1, u);
        y2 = M.retr(y1, u, (1/K)*t);
        y0 = y1;
        y1 = y2;
    end
    g(3, k) = M.dist(M.exp(x, v, t), y1);
end

%%
loglog(h, g(1, :), h, g(2, :), h, g(3, :));
tilte('Distance to geodesic');
legend('Single retraction', 'Multi retraction, isotransp', 'Multi retraction, transp');
xlabel('Step size');
