function Test_synchronization


n = 3;
m = 10;
A = randn(m, n)
SOn = rotationsfactory(n);
B = A*SOn.rand()+.1*randn(m, n)
C = A*SOn.rand()+.1*randn(m, n)

% Create the problem structure.
manifold = rotationsfactory(n, 2);
problem.M = manifold;

% Define the problem cost function and its gradient.
problem.cost = @cost;
function f = cost(R)
    RA = R(:, :, 1); RB = R(:, :, 2);
    f = norm(A*RA-B*RB, 'fro')^2 + norm(A*RA-C, 'fro')^2 ...
      + norm(B*RB-C, 'fro')^2;
end

problem.grad = @(R) manifold.egrad2rgrad(R, grad(R));
function G = grad(R)
    RA = R(:, :, 1); RB = R(:, :, 2);
    G = zeros(n, n, 2);
    G(:, :, 1) = -2*A'*(B*RB+C);
    G(:, :, 2) = -2*B'*(A*RA+C);
end

% Solve.
opts.tolgradnorm = 1e-14;
opts.rho_regularization = 1e1;
[R, Rcost, info1] = trustregions(problem, [], opts); %#ok<ASGLU>
% [R Rcost info2] = conjugategradient(problem, [], opts);

% Display some statistics.
subplot(2, 1, 1);
handle1 = semilogy([info1.time], [info1.gradnorm], '.-');
pbaspect([1.6 1 1]);
box off;
title('gradnorm');
subplot(2, 1, 2);
handle2 = plot([info1.time], [info1.cost], '.-');
pbaspect([1.6 1 1]);
box off;
title('cost');

set(handle1, 'Color', [223 186 105]/255);
set(handle1, 'MarkerSize', 25);
set(handle1, 'LineWidth', 3);
set(handle2, 'Color', [223 186 105]/255);
set(handle2, 'MarkerSize', 25);
set(handle2, 'LineWidth', 3);
% xlim([0 .1])
% ylim([1e-10 1e2]);
% set(gca, 'XTick', [0 .1]);
% set(gca, 'YTick', [1e-10 1e-6 1e-2 1e2]);


% hold on;
% 
% % options.Delta0 = 8;
% % options.Delta_bar = 20;
% options.tolgradnorm = 1e-8;
% M = [eye(n) proj(A'*B) proj(A'*C) ; proj(B'*A) eye(n) proj(B'*C) ; proj(C'*A) proj(C'*B) eye(n)]
% [V, D] = eig(M);
% V = real(V);
% R0 = zeros(n, n, 3);
% R0(:, :, 1) = proj(V(1:n, 1:n));
% R0(:, :, 2) = proj(V((n+1):2*n, 1:n));
% R0(:, :, 3) = proj(V((2*n+1):3*n, 1:n));
% R0(:, :, 1) = R0(:, :, 1) * R0(:, :, 3)';
% R0(:, :, 2) = R0(:, :, 2) * R0(:, :, 3)';
% R0(:, :, 3) = R0(:, :, 3) * R0(:, :, 3)';
% R0 = R0(:, :, 1:2);
% [R Rcost info] = trustregions(problem, R0, options);
% 
% semilogy([info.iter], [info.gradnorm], 'm.-');

end

    function Q = proj(X)
        n = size(X, 1);
        [U, ~, V] = svd(X);
        J = diag([ones(1, n-1), sign(det(U*V'))]);
        Q = U*J*V';
    end