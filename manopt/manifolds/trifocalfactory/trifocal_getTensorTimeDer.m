function dTens = trifocal_getTensorTimeDer(X,dX)
% function dTens = trifocal_getTensorTimeDer(X,dX)
%
% compute dT/dt from X and dX/dt
%
% See also trifocal_getTensor

k = size(X,3);

Tens = zeros([3 3 3 k]);
dTens = zeros([3 3 3 k]);

for j=1:k 
    
        R1 = X(:,1:3,j);
        R2 = X(:,4:6,j);
        R3 = X(:,7:9,j);
        T12 = X(:,10,j);
        T13 = X(:,11,j);

        dR1  = dX(:,1:3,j);
        dR2  = dX(:,4:6,j);
        dR3  = dX(:,7:9,j);
        dT12 = dX(:,10,j);
        dT13 = dX(:,11,j);

        Q31 = R3'*R1;
        Q21 = R2'*R1;


        Tens(:,:,1) = R2'*T12*(Q31(:,1))'-Q21(:,1)*( R3'*T13)';
        Tens(:,:,2) = R2'*T12*(Q31(:,2))'-Q21(:,2)*( R3'*T13)';
        Tens(:,:,3) = R2'*T12*(Q31(:,3))'-Q21(:,3)*( R3'*T13)';


        I = eye(3);

        for i=1:3
        dTens(:,:,i,j) = R2'*T12*I(:,i)'*dR1'*R3 - R2'*dR1*I(:,i)*T13'*R3 + ...
                          dR2'*R2*Tens(:,:,i) + Tens(:,:,i)*R3'*dR3 + ...
                          R2'*dT12*I(:,i)'*R1'*R3-R2'*R1*I(:,i)*dT13'*R3;
        end

end

end
