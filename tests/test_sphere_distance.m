function test_sphere_distance
n = 5;
M = spherefactory(n);

% Pick a deterministic point&vector or a random pair.
% Advantage of deterministic: reduces errors due to M.exp()
x = M.rand();
u = M.randvec(x); % has unit norm
% x = [1 ; zeros(n-1, 1)];
% u = [0 ; 1 ; zeros(n-2, 1)];


% The three distances we compare
dist1 = @mydist; % new, see below -- this is the best one
dist2 = @(x, y) 2*asin(.5*norm(x-y, 2)); % <- this is already really good!
dist3 = M.dist;

g1 = @(t) dist1(x, M.exp(x, u, t)); % new distance
g2 = @(t) dist2(x, M.exp(x, u, t)); % asin distance
g3 = @(t) dist3(x, M.exp(x, u, t)); % manopt distance

% Pick which range of distances to explore
h = logspace(-20, log10(pi), 1001); % to explore full range
% h = logspace(log10(pi)-.01, log10(pi), 1001); % to explore very close to maximal distance

% Maje sure the last distance is the IEEE representation of pi.
h(end) = pi;


z = zeros(size(h)); for k = 1 : length(h), z(k) = g1(h(k)); end
w = zeros(size(h)); for k = 1 : length(h), w(k) = g2(h(k)); end
v = zeros(size(h)); for k = 1 : length(h), v(k) = g3(h(k)); end

loglog(h, abs(z-h)./h, '.'); hold all;
loglog(h, abs(w-h)./h, 'o');
loglog(h, abs(v-h)./h, 's'); hold off;

legend('new', 'asin', 'old manopt (acos or chordal) -- unless using updated manopt');

xlim([min(h),max(h)])

xlabel('Geodesic distance between x and y');
title('Relative error on computation of geodesic distance');


%%

dasin = @(x)  1./sqrt(1-x.^2);
dacos = @(x) -1./sqrt(1-x.^2);

dists = linspace(0, 2, 1001);
inner = (1-.5*dists.^2);

sensitivity_acos = 2*abs(dacos(inner) .* inner);

sensitivity_asin = 3*abs(dasin(.5.*dists) .* .5.*dists);

figure;
semilogy(dists, abs(sensitivity_acos-1)./dists, dists, abs(sensitivity_asin-1)./dists);
legend('acos', 'asin');

% The sensitivities cross when dist(x, y) = .8944; which is an inner of .6
% acos is better if dist(x, y) > cross, and asin is better otherwise.
% Both methods are in trouble if x is almost -y;
% If x almost opposite y, could reverse one of them and use asin?

end

function d = mydist(x, y)

    chordal = norm(x-y);
    
    % The asin formula is really good, except if x and y are almost
    % antipodal. In that case, it's best to change the sign of one of the
    % vectors and to compute the distance between (x, -y) : you get the
    % distance between x and y by comparing to pi.
    if chordal > 1.9 %2 - eps(2)
%         d = pi;
        d = pi - mydist(x, -y);
        return;
    end

    if chordal < 0.8944
        d = 2*asin(.5*chordal);
    else
        inner = x'*y;
        d = acos(inner);
    end
    
end