function rhess = trifocal_ehessT2rhess(X,egradT,ehessT,dX)
% rhess = essential_ehessT2rhess(X, egradT, ehessT, dX)
% Converts the Hessian in trifocal tensor T to the Riemannian Hessian in X.
%
% egradT is the function handle for the gradient in T.
% ehessT is the function handle for the Hessian in T.
% dX is the search direction in the space of X.
%
% The output is the Riemannian Hessian in the space of X
%
% See also: trifocalfactory trifocal_egradT2egrad


k = size(X,3);

T = trifocal_getTensor(X);

G = egradT(T);

dT = trifocal_getTensorTimeDer(X,dX);
dG = ehessT(T, dT);

M = trifocalfactory(1);
rhess = zeros([3 11 k]);
for i = 1:k
    rhess(:,:,i) = trifocal_ehessT2rhessSingle(X(:,:,i),G(:,:,:,i),dX(:,:,i),dG(:,:,:,i),T(:,:,:,i));
    rhess(:,:,i) =  M.proj( X(:,:,i),rhess(:,:,i));
end


function Xi = trifocal_ehessT2rhessSingle(X,G,dX,dG,T)  

    
e1 =[1;0;0];
e2 =[0;1;0];
e3 =[0;0;1];

R1 = X(:,1:3);
R2 = X(:,4:6);
R3 = X(:,7:9);
T12 = X(:,10);
T13 = X(:,11);

dR1 = dX(:,1:3);
dR2 = dX(:,4:6);
dR3 = dX(:,7:9);
dT12 = dX(:,10);
dT13 = dX(:,11);

Xia =zeros(3,11);

Xia(:,1:3) =   2*((R3*dG(:,:,1)'*R2'*T12 - R2*dG(:,:,1)*R3'*T13)*e1' + ...
                  (R3*dG(:,:,2)'*R2'*T12 - R2*dG(:,:,2)*R3'*T13)*e2' + ...
                  (R3*dG(:,:,3)'*R2'*T12 - R2*dG(:,:,3)*R3'*T13)*e3' );

Xia(:,4:6) =  2*R2*(T(:,:,1) *dG(:,:,1)' + T(:,:,2)* dG(:,:,2)' + T(:,:,3) *dG(:,:,3)');
Xia(:,7:9) =  2*R3*(T(:,:,1)'*dG(:,:,1)  + T(:,:,2)'*dG(:,:,2)  + T(:,:,3)'*dG(:,:,3) );

Xia(:,10)	 =     R2*dG(:,:,1)* R3'*R1*e1 + R2*dG(:,:,2)* R3'*R1*e2 + R2*dG(:,:,3)* R3'*R1*e3 ;          
Xia(:,11)	 = - ( R3*dG(:,:,1)'*R2'*R1*e1 + R3*dG(:,:,2)'*R2'*R1*e2 + R3*dG(:,:,3)'*R2'*R1*e3 );      
           



Xib =zeros(3,11); 

Xib(:,1:3) =  - multisym( R3*G(:,:,1)'*R2'*T12*e1'*R1')*dR1 - ...   % dR1 with dR1
                multisym( R3*G(:,:,2)'*R2'*T12*e2'*R1')*dR1 - ...   % dR1 with dR1
                multisym( R3*G(:,:,3)'*R2'*T12*e3'*R1')*dR1  ...   % dR1 with dR1
              + (multisym(R2*G(:,:,1)*R3'*T13*e1'*R1'))*dR1 +...      % dR1 with dR1
                (multisym(R2*G(:,:,2)*R3'*T13*e2'*R1'))*dR1 +... 
                (multisym(R2*G(:,:,3)*R3'*T13*e3'*R1'))*dR1 +... 
             +1*(dR3*G(:,:,1)'*R2'*T12*e1' +...                % dR1 with dR3
                 dR3*G(:,:,2)'*R2'*T12*e2' +...
                 dR3*G(:,:,3)'*R2'*T12*e3' )...
             -1*(  R2*G(:,:,1)*dR3'*T13*e1'+... 
                 R2*G(:,:,2)*dR3'*T13*e2'+... 
                 R2*G(:,:,3)*dR3'*T13*e3')+ ...
             1*(  (R3*G(:,:,1)'*R2'*dT12*e1'-R2*G(:,:,1)*R3'*dT13*e1' ) + ... % dR1 with dT  
                  (R3*G(:,:,2)'*R2'*dT12*e2'-R2*G(:,:,2)*R3'*dT13*e2' ) + ...
                  (R3*G(:,:,3)'*R2'*dT12*e3'-R2*G(:,:,3)*R3'*dT13*e3' ) )+...
             +1*(    R3*G(:,:,1)'*dR2'*T12*e1' + ...            % dR2 with dR1
                     R3*G(:,:,2)'*dR2'*T12*e2'  + ... 
                     R3*G(:,:,3)'*dR2'*T12*e3' ) ... 
            -1*( (e1*T13'*R3*G(:,:,1)'*dR2')' + ...
                 (e2*T13'*R3*G(:,:,2)'*dR2')' + ...
                 (e3*T13'*R3*G(:,:,3)'*dR2')' );      
             
             

Xib(:,4:6) =  - multisym( R2*T(:,:,1)*G(:,:,1)'*R2')*dR2 - ...       % dR2 with dR2
                multisym( R2*T(:,:,2)*G(:,:,2)'*R2')*dR2 - ...      % dR2 with dR2
                multisym( R2*T(:,:,3)*G(:,:,3)'*R2')*dR2 ...         % dR2 with dR2
             +1*(    T12*e1'*dR1'*R3*G(:,:,1)' + ...            % dR2 with dR1
                     T12*e2'*dR1'*R3*G(:,:,2)' + ... 
                     T12*e3'*dR1'*R3*G(:,:,3)' ) ... 
            -1*(dR1*e1*T13'*R3*G(:,:,1)' + ...
                dR1*e2*T13'*R3*G(:,:,2)' + ...
                dR1*e3*T13'*R3*G(:,:,3)' ) ...
             +1*(    T12*e1'*R1'*dR3*G(:,:,1)' + ...            % dR2 with dR3
                     T12*e2'*R1'*dR3*G(:,:,2)' + ... 
                     T12*e3'*R1'*dR3*G(:,:,3)' ) ... 
              -1*(R1*e1*T13'*dR3*G(:,:,1)' + ...
                  R1*e2*T13'*dR3*G(:,:,2)' + ...
                  R1*e3*T13'*dR3*G(:,:,3)' ) ...
               +1*(  (dT12*e1'*R1'*R3-R1*e1*dT13'*R3)*G(:,:,1)' +... % dR2 with dT
                     (dT12*e2'*R1'*R3-R1*e2*dT13'*R3)*G(:,:,2)' +...
                     (dT12*e3'*R1'*R3-R1*e3*dT13'*R3)*G(:,:,3)');
            

            
Xib(:,7:9) =  - multisym( R3*T(:,:,1)'*G(:,:,1)*R3')*dR3 - ...      % dR3 with dR3
                multisym( R3*T(:,:,2)'*G(:,:,2)*R3')*dR3 - ...      % dR3 with dR3
                multisym( R3*T(:,:,3)'*G(:,:,3)*R3')*dR3 + ...          
               1*(  (R1*e1*dT12'*R2-dT13*e1'*R1'*R2)*G(:,:,1) +...   % dR3 with dT 
                    (R1*e2*dT12'*R2-dT13*e2'*R1'*R2)*G(:,:,2) +...  
                    (R1*e3*dT12'*R2-dT13*e3'*R1'*R2)*G(:,:,3) ) +...
              +1*(    dR1*e1*T12'*R2*G(:,:,1) + ...                % dR1 with dR3
                    dR1*e2*T12'*R2*G(:,:,2)   + ...
                    dR1*e3*T12'*R2*G(:,:,3)   ) ...
             -1*( T13*e1'*dR1'*R2*G(:,:,1) +... 
                  T13*e2'*dR1'*R2*G(:,:,2) +... 
                  T13*e3'*dR1'*R2*G(:,:,3) )+...
             +1*(    (G(:,:,1)'*dR2'*T12*e1'*R1')' + ...            % dR2 with dR3
                     (G(:,:,2)'*dR2'*T12*e2'*R1')'  + ... 
                     (G(:,:,3)'*dR2'*T12*e3'*R1')'  ) ... 
              -1*( (G(:,:,1)'*dR2'*R1*e1*T13')' + ...
                  (G(:,:,2)'*dR2'*R1*e2*T13')' + ...
                  (G(:,:,3)'*dR2'*R1*e3*T13')' ) ;
            

 a1 = e1'*R1'*R3*G(:,:,1)'*R2'*T12 + e2'*R1'*R3*G(:,:,2)'*R2'*T12 +  e3'*R1'*R3*G(:,:,3)'*R2'*T12;
 a2 = e1'*R1'*R2*G(:,:,1)*R3'*T13 +  e2'*R1'*R2*G(:,:,2)*R3'*T13   + e3'*R1'*R2*G(:,:,3)*R3'*T13;
  

Xib(:,10:11) =  (a2-a1)*[dT12 dT13]+... %dT with dT  
                [(e1'*dR1'*R3*G(:,:,1)'*R2')'    -(e1'*dR1'*R2*G(:,:,1)*R3')']+... % dR1 with dT
                [(e2'*dR1'*R3*G(:,:,2)'*R2')'    -(e2'*dR1'*R2*G(:,:,2)*R3')']+...
                [(e3'*dR1'*R3*G(:,:,3)'*R2')'    -(e3'*dR1'*R2*G(:,:,3)*R3')'] + ... 
                [ (e1'*R1'*R3*G(:,:,1)'*dR2')'   -R3*G(:,:,1)'*dR2'*R1*e1  ] + ...% dR2 with dT
                [ (e2'*R1'*R3*G(:,:,2)'*dR2')'   -R3*G(:,:,2)'*dR2'*R1*e2  ] + ... 
                [ (e3'*R1'*R3*G(:,:,3)'*dR2')'   -R3*G(:,:,3)'*dR2'*R1*e3  ] + ...
                [R2*G(:,:,1)*dR3'*R1*e1   -(e1'*R1'*R2*G(:,:,1)*dR3')'] + ... % dR3 with dT 
                [R2*G(:,:,2)*dR3'*R1*e2   -(e2'*R1'*R2*G(:,:,2)*dR3')'] + ...
                [R2*G(:,:,3)*dR3'*R1*e3   -(e3'*R1'*R2*G(:,:,3)*dR3')'];


      
Xib(:,1:9) = 2*Xib(:,1:9);
Xi = Xia + Xib;
Xi(3,10:11) = 0;

end

end