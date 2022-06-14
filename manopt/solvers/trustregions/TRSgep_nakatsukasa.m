function [x,lam1] = TRSgep_nakatsukasa(A,a,B,Del)
% Solves the trust-region subproblem by a generalized eigenproblem without
% iterations
% 
% minimize (x^TAx)/2+ ax
% subject to x^TBx <= Del^2
%
% A: nxn symmetric, a: nx1 vector
% B: nxn symmetric positive definite
% 
% Yuji Nakatsukasa, 2015
% Revised by Nikitas Rontsis, December 2018

n = size(A,1);
if issparse(B)
MM1 = [sparse(n,n) B;B sparse(n,n)];
else
MM1 = [zeros(n) B;B zeros(n)];    
end
tolhardcase = 1e-4; % tolerance for hard-case

p1 = pcg(A,-a,1e-12, 500); % possible interior solution
if norm(A*p1+a)/norm(a)<1e-5,
if p1'*B*p1>=Del^2, p1 = nan;
end
else
    p1 = nan;
    end

% This is the core of the code
[V,lam1] = eigs(@(x)MM0timesx(A,B,a,Del,x),2*n,-MM1,1,'lr'); 

    if norm(real(V)) < 1e-3 %sometimes complex
        V = imag(V);    else        V = real(V);
    end

    lam1 = real(lam1);
    x = V(1:length(A)); % this is parallel to soln
    normx = sqrt(x'*(B*x));         
    x = x/normx*Del; % in the easy case, this naive normalization improves accuracy
    if x'*a>0, x = -x; end % take correct sign
    
if normx < tolhardcase % hard case
%disp(['hard case!',num2str(normx)])
x1 = V(length(A)+1:end);
alpha1 = lam1;
Pvect = x1;  %first try only k=1, almost always enough
x2 = pcg(@(x)pcgforAtilde(A,B,lam1,Pvect,alpha1,x),-a,1e-12,500);
if norm((A+lam1*B)*x2+a)/norm(a)>tolhardcase, % large residual, repeat
    for ii = 3*[1:3]
    [Pvect,DD] = eigs(A,B,ii,'sa');
    x2 = pcg(@(x)pcgforAtilde(A,B,lam1,Pvect,alpha1,x),-a,1e-8,500);    
    if norm((A+lam1*B)*x2+a)/norm(a) < tolhardcase, break, end
    end
end

Bx = B*x1; Bx2 = B*x2; aa = x1'*(Bx); bb = 2*x2'*Bx; cc = (x2'*Bx2-Del^2); 
alp = (-bb+sqrt(bb^2-4*aa*cc))/(2*aa); %norm(x2+alp*x)-Delta
x = x2+alp*x1;
end

% choose between interior and boundary 

if sum(isnan(p1))==0,
if (p1'*A*p1)/2+a'*p1 < (x'*A*x)/2+a'*x, 
    x = p1; lam1 = 0;
end
end
end



function [y] = MM0timesx(A,B,g,Delta,x)
% MM0 = [-B A;
%         A -g*g'/Delta^2];
n = size(A,1); 
x1 = x(1:n); x2 = x(n+1:end);
y1 = -B*x1 + A*x2;
y2 = A*x1-g*(g'*x2)/Delta^2;
y = [y1;y2];
end

function [y] = pcgforAtilde(A,B,lamA,Pvect,alpha1,x)

[n,m] = size(Pvect);
y = A*x+lamA*(B*x);

for i=1:m
    y = y+(alpha1*(x'*(B*Pvect(:,i))))*(B*Pvect(:,i));
end
end