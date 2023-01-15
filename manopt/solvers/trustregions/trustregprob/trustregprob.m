function [x,fval] = trustregprob(Q,b,w,doEquality)
%A routine for analytically minimizing quadratic functions subject to constraints on
%the l2-norm of the variables. The problem is of a form commonly encountered as a 
%sub-problem in trust region algorithms, but undoubtedly has other applications
%as well. 
%
%USAGE:
%   
%     [xmin,Jmin] = trustregprob(Q,b,w) 
%     [xmin,Jmin] = trustregprob(Q,b,w,doEquality) 
%   
%When doEquality=true (the default), the routine solves, 
%   
%     minimize J(x) = x.'*Q*x/2-dot(b,x) such that ||x|| = w 
%   
%where ||x|| is the l2-norm of x. The variables returned xmin, Jmin are the
%minimizing x and its objective function value J(x).
%
%When doEquality=false, the routine solves instead subject to ||x|| <= w . 
%   
%Q is assumed symmetric, but not necessarily positive semi-definite. In other words, the 
%objective function J(x) is potentially non-convex. Since the solution is based on 
%eigen-decomposition, it is appropriate mainly for Q not too large. If multiple solutions
%exist, only one solution is returned.


N=size(Q,1);
if numel(b)~=N, error 'Q and b are of mismatched sizes'; end

if ~exist('doEquality','var')
   doEquality=true; 
end

if w<0, error 'Argument ''w'' must be non-negative.'; end

if ~w
    x=zeros(size(b));x=x(:);
    fval=0;
    return
end


[V,H]=eigs((Q+Q)/2);
h=diag(H);

posdefIneq=all(h>=0) & ~doEquality;

if ~isreal(h) || ~isreal(V)
    disp 'Q not symmetric?'
    keyboard; 
end

if ~any(b)
    
    if posdefIneq
   
      x=b; %all zeros  
      fval=0;
    else
     [lambda,imin] = min(h);
      
      x=w*V(:,imin);
      fval=w^2*lambda/2;
    end
    
    
    
elseif posdefIneq
    
    Vt=V.';
    g=Vt*b;
    x=g./h;
    if norm(x)<w
  
        if nargout>1,
          fval=(x.'*(h.*x))/2 - dot(g,x);
        end

        x=V*x;

    else
        
        [x,fval]=  trustregprob(spdiags(h,0),g,w);
        
    end
    
else   
    
    %diagonalize
    Vt=V.';
    g=Vt*b;
    
    
    hmin=min(h);
    
    idx=logical(g);
    gnz=g(idx);
    hnz=h(idx);
    
    [him,im]=min(hnz);
     gim=gnz(im);
  
    fun=@(lambda) 1- w/norm( gnz./(hnz-lambda) );
    

    fhigh=fun(hmin);
    if fhigh>0
        
            ub=hmin;
    
            i=0;
            lb=ub-10^i;
            while fun(lb)>=0

                i=i+1;
                lb=ub-10^i;

            end

   
                lambda=fzero(fun,[lb,ub]);

    else
       
        lambda=hmin;
        
    end
    

    hl=h-lambda;
    x=zeros(size(g));
    idx=hl>0;
    x(idx)=g(idx)./hl(idx); 
    
     cc=w^2-norm(x)^2;
     nidx=~idx;
     nn=sum(nidx);  
     x(nidx)=sqrt(cc/nn);    

    
    if nargout>1,
      fval=(x.'*(h.*x))/2 - dot(g,x);
    end
    
    x=V*x;
    
end
