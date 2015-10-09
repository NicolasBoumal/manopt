function Xi = trifocal_egradT2egrad(X,egradT)
% function Xi = trifocal_egradT2egrad(X,egradT)
% Converts the gradient in trifocal tensor T to the Riemannian Hessian in X.
%
% function rhess = trifocal_ehessT2rhess(X,egradT,ehessT,dX)
%
% egradT is the function handle for the gradient in T.
% ehessT is the function handle for the Hessian in T.
% dX is the search direction in the space of X.
%
% The output is the Euclidean gradient in the space of X
%
% See also: trifocalfactory trifocal_ehessT2rhess

k  = size(X,3);
Xi = zeros([3 11 k]);


G= egradT(trifocal_getTensor(X));

for i=1:k
    Xi(:,:,i) =  trifocal_egradT2egradSingle(X(:,:,i),G(:,:,:,i));
end


function Xi = trifocal_egradT2egradSingle(X,G)

T = trifocal_getTensor(X);

e1 =[1;0;0];
e2 =[0;1;0];
e3 =[0;0;1];

R1 = X(:,1:3);
R2 = X(:,4:6);
R3 = X(:,7:9);
T12 = X(:,10);
T13 = X(:,11);

Xi =zeros(3,11);

Xi(:,1:3) =   2*((R3*G(:,:,1)'*R2'*T12 - R2*G(:,:,1)*R3'*T13)*e1' + ...
                 (R3*G(:,:,2)'*R2'*T12 - R2*G(:,:,2)*R3'*T13)*e2' + ...
                 (R3*G(:,:,3)'*R2'*T12 - R2*G(:,:,3)*R3'*T13)*e3' );

Xi(:,4:6) =   2*R2*(T(:,:,1)*G(:,:,1)' + T(:,:,2)*G(:,:,2)' + T(:,:,3)*G(:,:,3)');

Xi(:,7:9) =  2*R3*(T(:,:,1)'*G(:,:,1) +T(:,:,2)'*G(:,:,2) +T(:,:,3)'*G(:,:,3));

xi41 =  R2*G(:,:,1)*R3'*R1*e1 + ...
        R2*G(:,:,2)*R3'*R1*e2 + ...
        R2*G(:,:,3)*R3'*R1*e3 ;
            
    
xi42 = - ( R3*G(:,:,1)'*R2'*R1*e1 + ...
           R3*G(:,:,2)'*R2'*R1*e2 + ...
           R3*G(:,:,3)'*R2'*R1*e3 );      
            
Xi(:,10:11) = [xi41 xi42];

end

end
