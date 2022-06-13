% https://groups.google.com/g/manopttoolbox/c/KwdpyLiPUBw/m/aS-Yjq-pAwAJ


%% Test exponential
M = rotationsfactory(3);
X = M.rand();
V = M.randvec(X);
t = logspace(-16, 3, 501);

e = NaN(size(t));
f = NaN(size(t));
for k = 1 : numel(t)

    Y1 = X*expm(t(k)*V);
    Y2 = X*expm_SO3(t(k)*V);

    e(k) = norm(Y1-Y2, 'fro') ./ sqrt(3);
    f(k) = norm(Y2-X,  'fro') ./ sqrt(3);  % make sure we're actually moving, not just returning X below some tolerance

end

loglog(t, e, '.', t, f, '.');

any(isnan(e)) % should be false
min(e) % should be zero or close


%% Test logarithm
M = rotationsfactory(3);
X = M.rand();
V = M.randvec(X);
t = logspace(-16, 3, 501);

e = NaN(size(t));
f = NaN(size(t));
for k = 1 : numel(t)

    e(k) = norm((t(k)*V) - logm_SO3(X.'*(X*expm_SO3(t(k)*V))), 'fro') ;%./ norm(t(k)*V, 'fro');
    f(k) = norm((t(k)*V) - logm(X.'*(X*expm(t(k)*V))), 'fro') ;%./ norm(t(k)*V, 'fro');

end

loglog(t, e, '.', t, f, '.');
legend('logm_SO3', 'logm', 'Location', 'northwest');

any(isnan(e)) % should be false
min(e) % should be zero or close

%% Speed test expm

t = randn(1);
timeit(@() X*expm(t*V))
timeit(@() X*expm_SO3(t*V))  % 10x faster, it seems

%% Speed test logm

Y = M.rand();
timeit(@() logm(Y))
timeit(@() logm_SO3(Y))  % 100x faster ??


%%


function phi = logm_SO3(R)
    t = trace(R);
    norm_t = real(acos((t - 1)/2));
    if norm_t > 0 % could fail even when trace(R) < 3, because sensitive
        q = .5*norm_t/sin(norm_t);
    else
        q = .5;   % even with this, phi (below) could be nonzero
    end
    phi = q * [R(3, 2) - R(2, 3); R(1, 3) - R(3, 1); R(2, 1) - R(1, 2)];
    phi = [0 -phi(3) phi(2); phi(3) 0 -phi(1); -phi(2) phi(1) 0];
end

function R = expm_SO3(phi)
    phi_vee = [-phi(2, 3); phi(1, 3); -phi(1, 2)];
    norm_phi_vee = norm(phi_vee);
    if norm_phi_vee > 0
        q1 = sin(norm_phi_vee)/norm_phi_vee;
        r = norm_phi_vee / 2;
        q2 = (sin(r)/r).^2 / 2;
        R = eye(3) + q1*phi + q2*phi^2;
    else
        R = eye(3);
    end
end
