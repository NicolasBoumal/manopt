% https://groups.google.com/g/manopttoolbox/c/KwdpyLiPUBw/m/aS-Yjq-pAwAJ

M = rotationsfactory(3);
X = M.rand();
V = M.randvec(X);
t = logspace(-16, 3, 501);

%% Test exponential
e = NaN(size(t));
f = NaN(size(t));
for k = 1 : numel(t)

    Y1 = M.exp(X, V, t(k));
    Y2 = X*expm_SO3(t(k)*V);

    e(k) = norm(Y1-Y2, 'fro') ./ sqrt(3);
    f(k) = norm(Y2-X,  'fro') ./ sqrt(3);  % make sure we're actually moving, not just returning X below some tolerance

end

loglog(t, e, '.', t, f, '.');

any(isnan(e)) % should be false
min(e) % should be zero or close


%% Test logarithm
e = NaN(size(t));
f = NaN(size(t));
for k = 1 : numel(t)

    e(k) = norm((t(k)*V) - logm_SO3(X.'*(X*expm_SO3(t(k)*V))), 'fro') ;%./ norm(t(k)*V, 'fro');
    f(k) = norm((t(k)*V) - M.log(X, (M.exp(X, V, t(k)))), 'fro') ;%./ norm(t(k)*V, 'fro');

end

loglog(t, e, '.', t, f, '.');

any(isnan(e)) % should be false
min(e) % should be zero or close

%% Speed test expm

t = randn(1);
timeit(@() M.exp(X, V, t))
timeit(@() X*expm_SO3(t*V))  % 10x faster, it seems

%% Speed test logm

Y = M.rand();
timeit(@() M.log(X, Y))
timeit(@() logm_SO3(X.'*Y))  % 1000x faster ??


%%


function phi = logm_SO3(R)
    t = trace(R);
    norm_t = real(acos((t - 1)/2));
    if norm_t > 0 % could fail even when trace(R) < 3, because sensitive
        q = .5*norm_t/sin(norm_t);
    else % if norm_t = 0 numerically, we get better accuracy with Taylor
        q = polyval([1/280, 1/60, 1/12, 1/2], 3-t); %.5 + (3-t)/12 + (3-t).^2/60 + (3-t).^3/280; %
    end
    phi = q * [R(3, 2) - R(2, 3); R(1, 3) - R(3, 1); R(2, 1) - R(1, 2)];
    phi = [0 -phi(3) phi(2); phi(3) 0 -phi(1); -phi(2) phi(1) 0];
%    phi = logm(R);
end

function R = expm_SO3(phi)
    phi_vee = [-phi(2, 3); phi(1, 3); -phi(1, 2)];
    norm_phi_vee = norm(phi_vee);
    q1 = sinxoverx(norm_phi_vee);
    q2 = sinxoverx(norm_phi_vee/2).^2 / 2;
    R = eye(3) + q1*phi + q2*phi^2;
%    R = expm(phi);
end
