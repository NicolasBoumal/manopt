
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
for k = 1 : numel(t)

    e(k) = norm((t(k)*V) - logm_SO3(X.'*(X*expm_SO3(t(k)*V))), 'fro') ./ norm(t(k)*V, 'fro');

end

loglog(t, e, '.');

any(isnan(e)) % should be false
min(e) % should be zero or close


%%


function phi = logm_SO3(R)
    norm_t = acos((trace(R) - 1)/2);
    phi = norm_t/(2*sin(norm_t))*[R(3,2) - R(2,3); R(1,3) - R(3,1); R(2,1) - R(1,2)];
    phi = [0 -phi(3) phi(2); phi(3) 0 -phi(1); -phi(2) phi(1) 0];
%    phi = logm(R);
end

function R = expm_SO3(phi)
    I = eye(3);
    phi_vee = [-phi(2,3); phi(1,3); -phi(1,2)];
    norm_phi_vee = norm(phi_vee);
    R = I + sin(norm_phi_vee)/norm_phi_vee*phi + ((1 - cos(norm_phi_vee))/norm_phi_vee^2)*phi^2;
%    R = expm(phi);
end
