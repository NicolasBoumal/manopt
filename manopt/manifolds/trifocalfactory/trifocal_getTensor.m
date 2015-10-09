function T = trifocal_getTensor(X)
%function T = trifocal_getTensor(X)
%
% Conversion from 3x11xk representation to 3x3x3xk tensor
%
% See also: trifocalfactory

k = size(X,3); 

T= zeros(3,3,3,k);

for i=1:k

    R1 = X(:,1:3,i);
    R2 = X(:,4:6,i);
    R3 = X(:,7:9,i);
    T12 = X(:,10,i);
    T13 = X(:,11,i);

    Q31 = R3'*R1;
    Q21 = R2'*R1;

    T(:,:,1,i) = R2'*T12*(Q31(:,1))'-Q21(:,1)*(R3'*T13)';
    T(:,:,2,i) = R2'*T12*(Q31(:,2))'-Q21(:,2)*(R3'*T13)';
    T(:,:,3,i) = R2'*T12*(Q31(:,3))'-Q21(:,3)*(R3'*T13)';

end


end

