function [x, limitedbyTR, accurate] = TRSgep(A, a, Del)
% Solves trust-region subproblem by a generalized eigenvalue problem.
% 
% function [x, limitedbyTR] = TRSgep(A, a, Del)
% function [x, limitedbyTR, accurate] = TRSgep(A, a, Del)
% 
% This function returns a solution x to the following optimization problem:
% 
%     minimize .5*(x.'*A*x) + a.'*x
%     subject to x.'*x <= Del^2
% 
% The boolean 'limitedbyTR' is true if the solution would have been
% different absent the norm constraint. In that case, the norm of x is Del.
%
% Inputs:
%   A: nxn symmetric
%   a: nx1 vector, both real
%   Del: trust-region radius (positive real)
%
% If called with three outputs, then computationally expensive checks are
% run to verify the accuracy of the output. If the output appears to be
% globally optimal (as expected) within some demanding numerical
% tolerances, then 'accurate' is true; otherwise it is false.
%
% Code adapted from Yuji Nakatsukasa's code for the
% paper by Satoru Adachi, Satoru Iwata, Yuji Nakatsukasa, and Akiko Takeda
%
% Original code: https://people.maths.ox.ac.uk/nakatsukasa/codes/TRSgep.m
% Reference paper: https://epubs.siam.org/doi/abs/10.1137/16M1058200
%
% The authors kindly allowed us to include their code in Manopt under the 
% same license as Manopt.
%
% See also: trs_gep trs_tCG_cached trs_tCG trustregions

% This file is part of Manopt: www.manopt.org.
% Original author: Yuji Nakatsukasa, 2015.
% Contributors: Revised by Nikitas Rontsis, December 2018
% Change log:
%   VL June 29, 2022:
%       Modified original code to return limitedbyTR boolean and change
%       ellipsoid norm constraint to unweighted norm.
%   NB Aug. 19, 2022:
%       Comments + cosmetic changes.
%       Corrected determination of limitedbyTR.
%       Added support for input a = 0.
%       Clarified the logic around picking the Newton step or not.

    n = size(A, 1);

    % We set this flag to true iff the solution x we eventually return is
	% limited by the trust-region boundary.
    limitedbyTR = true;

    % Tolerance for hard-case.
    % If this triggers, then the solver works harder to check itself.
    tolhardcase = 1e-4;

    % If a is exactly zero, pcg (called below) abandons on the first
    % iteration. Instead, we give it a small input and re-check at the end.
    a_is_zero = (norm(a) == 0);
    if a_is_zero
        a = eps*randn(n, 1);
    end

    % Compute the Newton step p1 up to some accuracy.
    [p1, ~, relres] = pcg(A, -a, 1e-12, 500);

    % If the Newton step is computed accurately and it is in the trust
    % region, then it may very well be the solution to the TRS.
    % We make a note of it, and will re-check at the end.
    newton_step_may_be_solution = (relres < 1e-5 && (p1'*p1 <= Del^2));

    % This is the core of the code.
    MM1 = [sparse(n, n) speye(n) ; speye(n) sparse(n, n)];
    [V, lam1] = eigs(@(x) MM0timesx(A, a, Del, x), 2*n, -MM1, 1, 'lr');

    % Sometimes the output is complex.
    if norm(real(V)) < 1e-3
        V = imag(V);
    else
        V = real(V);
    end
    lam1 = real(lam1);

    % This is parallel to the solution:
    x = V(1:n);
    normx = norm(x);

    % In the easy case, this naive normalization improves accuracy.
    x = x/normx*Del;
    % Take the correct sign.
    if x'*a > 0
        x = -x;
    end
    
    % If we suspect a (numerically) hard case, work harder.
    if normx < tolhardcase
        x1 = V(n+1:end);
        alpha1 = lam1;
        Pvect = x1;
        % First try only k = 1, that is almost always enough
        [x2, ~] = pcg(@(x) pcgforAtilde(A, lam1, Pvect, alpha1, x), -a, ...
                                                               1e-12, 500);
        % If large residual, repeat
        if norm((A+lam1)*x2 + a) > tolhardcase*norm(a)
            for ii = [3, 6, 9]
                [Pvect, ~] = eigs(A, speye(n), ii, 'sa');
                [x2, ~] = pcg(@(x) pcgforAtilde(A, lam1, Pvect, alpha1, x), ...
                                                            -a, 1e-8, 500);    
                if norm((A+lam1)*x2 + a) < tolhardcase*norm(a)
                    break;
                end
            end
        end

        aa = x1'*x1;
        bb = 2*(x2'*x1);
        cc = x2'*x2 - Del^2;
        % Move to the boundary: set alp such that norm(x2+alp*x1) = Delta.
        % alp = (-bb + sqrt(bb^2 - 4*aa*cc))/(2*aa);
        alp = max(real(roots([aa, bb, cc])));
        x = x2 + alp*x1;
    end

    % If we suspected that the Newton step might be the solution to the
    % TRS, we compare it to the boundary solution we just computed and pick
    % the best one.
    if newton_step_may_be_solution
        if (p1'*A*p1)/2 + a'*p1 < (x'*A*x)/2 + a'*x
            x = p1;
            limitedbyTR = false;
        end
    end

    % If the input a was zero, then earlier in the code we replaced it with
    % a tiny random vector. Two things may have happened afterwards:
    % If A is positive definite, then the solution x is also a tiny vector.
    % In all likelihood, it did not hit the TR: then we know to replace x
    % with zero.
    % Otherwise, at least one eigenvalue of A is <= 0, and there exists a
    % solution on the boundary of the trust-region: that is what should
    % have been computed already, hence we do nothing.
    if a_is_zero && ~limitedbyTR
        x = zeros(n, 1);
    end


    % This is for debugging purposes only: it is expensive to run.
    % The code checks via a dual certificate that x is a global optimum,
    % up to some numerical tolerances. It also checks limitedbyTR.
    if nargout >= 3
        tol = 1e-13;
        mineig = min(eig(A));
        if norm(x) ~= 0
            % Estimate the dual variable for the norm constraint.
            mu = -(x'*(A*x + a))/(x'*x);
            % The vector x is optimal iff:
            %   norm(x) <= Del,
            %   M = A + mu*I is psd and mu >= 0,
            %   M*x + b = 0, and
            %   mu = 0 whenever we are not limited by TR.
            % We also need that limitedbyTR => norm(x) == Del.
            reltol = @(c) c + tol*max(1, c); % to check a <~ c with c >= 0.
            accurate = (norm(x) <= reltol(Del) && ...
                        max(0, -mineig) <= reltol(mu) && ...
                        all(abs(A*x+a + mu*x) <= reltol(abs(mu*x))) && ...
                        ( limitedbyTR || mu <= reltol(0)) && ...
                        (~limitedbyTR || Del <= reltol(norm(x))));
            if ~accurate
                keyboard;
            end
        else
            % The zero vector x is optimal iff a = 0 and A is psd.
            % Moreover, a solution x = 0 is clearly not limited by the TR.
            accurate = (norm(a) <= tol && mineig >= -tol && ~limitedbyTR);
        end
    end
    
end



function y = MM0timesx(A, g, Delta, x)
    % MM0 = [-Id A;
    %         A -g*g'/Delta^2];
    n = size(A, 1); 
    x1 = x(1:n);
    x2 = x(n+1:end);
    y1 = -x1 + A*x2;
    y2 = A*x1 - g*(g'*x2)/Delta^2;
    y = [y1 ; y2];
end

function y = pcgforAtilde(A, lamA, Pvect, alpha1, x)
    m = size(Pvect, 2);
    y = A*x + lamA*x;
    for ii = 1:m
        y = y + (alpha1*(x'*(Pvect(:, ii))))*(Pvect(:, ii));
    end
end
