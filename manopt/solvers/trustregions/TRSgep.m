function [x, limitedbyTR] = TRSgep(A, a, Del)
% Solves trust-region subproblem by a generalized eigenvalue problem.
% 
% function [x, limitedbyTR] = TRSgep(A, a, Del)
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

    n = size(A, 1);

    % We set this flag to true iff the solution x computed below is
	% limited by the trust-region boundary.
    limitedbyTR = false;

    MM1 = [sparse(n, n) speye(n) ; speye(n) sparse(n, n)];
    tolhardcase = 1e-4; % tolerance for hard-case

    [p1, ~] = pcg(A, -a, 1e-12, 500); % possible interior solution
    if norm(A*p1 + a) < 1e-5*norm(a)
        if p1'*p1 >= Del^2
            p1 = NaN;
        end
    else
        p1 = NaN;
    end

    % This is the core of the code
    [V, lam1] = eigs(@(x) MM0timesx(A, a, Del, x), 2*n, -MM1, 1, 'lr');

    if norm(real(V)) < 1e-3 % sometimes complex
        V = imag(V);
    else
        V = real(V);
    end
    
    lam1 = real(lam1);
    % This is parallel to the solution:
    x = V(1:n);
    normx = sqrt(x'*x);
    % In the easy case, this naive normalization improves accuracy
    x = x/normx*Del;
    % Take the correct sign
    if x'*a > 0
        x = -x;
    end
    
    % Hard case
    if normx < tolhardcase
        %disp(['hard case!', num2str(normx)])
        x1 = V(length(A)+1:end);
        alpha1 = lam1;
        Pvect = x1;
        % First try only k = 1, that is almost always enough
        [x2, ~] = pcg(@(x) pcgforAtilde(A, lam1, Pvect, alpha1, x), -a, ...
                                                               1e-12, 500);
        % If large residual, repeat
        if norm((A+lam1)*x2 + a) > tolhardcase*norm(a)
            for ii = 3*(1:3)
                [Pvect, ~] = eigs(A, speye(n), ii, 'sa');
                [x2, ~] = pcg(@(x) pcgforAtilde(A, lam1, Pvect, alpha1, x), ...
                                                            -a, 1e-8, 500);    
                if norm((A+lam1)*x2+a) < tolhardcase*norm(a)
                    break
                end
            end
        end

        aa = x1'*(x1);
        bb = 2*x2'*x1;
        cc = (x2'*x2 - Del^2); 
        alp = (-bb + sqrt(bb^2 - 4*aa*cc))/(2*aa); %norm(x2+alp*x) - Delta
        x = x2 + alp*x1;
    end

    % Choose between interior and boundary
    if sum(isnan(p1)) == 0
        if (p1'*A*p1)/2 + a'*p1 < (x'*A*x)/2 + a'*x
            x = p1;
            % lam1 = 0;
            % Specify we have a boundary solution
            limitedbyTR = true;
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
