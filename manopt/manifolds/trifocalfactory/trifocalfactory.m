function M = trifocalfactory(k)
% Manifold structure to optimize over the space of trifocal tensors.
%
% function M = trifocalfactory(k)
%
%
% Quotient representation of the trifocal manifold: deals with the
% representation of the space of M_rT of trifocal tensors. These are used in
% computer vision to represent the constraints between projected
% points and/or lines in three perspective views.
%
% The space is represented as the quotient ( (SO(3)^3 x S_2^3) / (H_pi x SO(2)) ).
% See the following references for details:
%
%   S. Leonardos, R. Tron, K. Daniilidis,
%  "A Metric Parametrization for Trifocal Tensors With Non-Colinear Pinholes"
%   IEEE Conference on Computer Vision and Pattern Recognition, 2015
%
% For computational purposes, each trifocal tensor is represented as a
% [3x11] matrix where the first three [3x3] blocks are rotation matrices 
% and the last [3x2] block is unit Frobenius norm matrix with zeros in the last row.
%
% The metric used is the one induced by the submersion of M_{rT} in SO(3)^3 x S_2^3.
%
% Tangent vectors are represented in the ambient space.
%
% By default, k = 1.
%
% See also essentialfactory, rotationsfactory, spherefactory

% Please cite the Manopt paper as well as the research paper:
%     @InProceedings{leonardos2015trifocal,
%       Title        = {A Metric Parametrization for Trifocal Tensors With Non-Colinear Pinholes},
%       Author       = {Leonardos, S. and Tron, R. and Daniilidis, K.},
%       Booktitle    = {IEEE Conference on Computer Vision and Pattern Recognition},
%       Year         = {2015},
%       Organization = {{IEEE CVPR}}
%     }



 % Optional parameters 
 if ~exist('k', 'var') || isempty(k)
     k = 1;
 end

 if k == 1
     M.name = @() sprintf('Quotient representation of the trifocal manifold');
 elseif k > 1 && k == round(k)
     M.name = @() sprintf('Product of %d quotient representations of the trifocal manifold', k);
 else
     error('k must be an integer no less than 1.');
 end

 M.dim = @() k*11;

 
 M.T = @trifocal_getTensor;
 function T = trifocal_getTensor(X)

    % Conversion from 3x11xk representation to 3x3x3xk tensor
    T= zeros(3,3,3,k);

    for i=1:k

        R1 = X(:,1:3,i);
        R2 = X(:,4:6,i);
        R3 = X(:,7:9,i);
        T12 = X(:,10,i);
        T13 = X(:,11,i);

        Q31 = R3'*R1;
        Q21 = R2'*R1;

        T(:,:,1,i) = R2'*T12*(Q31(:,1))'-Q21(:,1)*( R3'*T13)';
        T(:,:,2,i) = R2'*T12*(Q31(:,2))'-Q21(:,2)*( R3'*T13)';
        T(:,:,3,i) = R2'*T12*(Q31(:,3))'-Q21(:,3)*( R3'*T13)';

    end

end
 
 M.inner = @(x, d1, d2) .5*sum( multitrace( multiprod( multitransp(d1(:,1:9,:))  , d2(:,1:9,:)  )))+...
                           sum( multitrace( multiprod( multitransp(d1(:,10:11,:)), d2(:,10:11,:))));
 
 M.norm = @(x, d) sqrt( M.inner(x,d,d) );

 M.typicaldist = @() pi*sqrt(4*k); 


 M.proj = @tangentProjection;
 function HProjHoriz=tangentProjection(X,H)
     
    H(3,10:11,:) = 0;
     
    HProj = cat(2,...
               multiprod( X(:,1:3,:),multiskew( multiprod( multitransp(X(:,1:3,:)),H(:,1:3,:)))), ...
               multiprod( X(:,4:6,:),multiskew( multiprod( multitransp(X(:,4:6,:)),H(:,4:6,:)))), ...
               multiprod( X(:,7:9,:),multiskew( multiprod( multitransp(X(:,7:9,:)),H(:,7:9,:)))), ...
               H(:,10:11,:) - multiprod( reshape( multitrace(multiprod(multitransp(X(:,10:11,:)),H(:,10:11,:))),[1 1 k]),X(:,10:11,:)));
     
    HProjHoriz = HProj - vertproj(X, HProj);
      
 end

M.tangent = M.proj; 

 M.egrad2rgrad=@egrad2rgrad;
 function rgrad = egrad2rgrad(X, egrad)
     rgrad = M.proj(X, egrad);
 end


 M.ehess2rhess = @ehess2rhess;
 function rhess = ehess2rhess(X, egrad, ehess, U)
   
     G = egrad; 
     H = ehess; 
     
      connection =  -cat(2,...
             multiprod(p1(U), multisym(multiprod(multitransp(p1(X)),p1(G)))),...
             multiprod(p2(U), multisym(multiprod(multitransp(p2(X)),p2(G)))),...
             multiprod(p3(U), multisym(multiprod(multitransp(p3(X)),p3(G)))),...
             multiprod(p4(U), repmat( multitrace(multiprod(multitransp(p4(X)), p4(G))),[1 1 k])));
         
     rhess = M.proj(X,H + connection);
     
     
 end


 M.exp = @exponential;
 function Y = exponential(X, U, t)
     
     % U in the form R'*skew for rotations
     
     if nargin == 3
         U = t*U;
     end

     Y = zeros(size(X));
     
     Y(:,1:3,:) = multiprod(p1(X), rot3_exp( multiprod( multitransp(p1(X)),p1(U))));
     Y(:,4:6,:) = multiprod(p2(X), rot3_exp( multiprod( multitransp(p2(X)),p2(U))));
     Y(:,7:9,:) = multiprod(p3(X), rot3_exp( multiprod( multitransp(p3(X)),p3(U))));
     
     
     Y(:,10:11,:) = Sphere_exp(p4(X),p4(U));
     
 end

 M.retr = @exponential;

M.log = @logarithm; 

function U = logarithm(X, Y)  

  U = zeros(3,11,k);
  
  for i=1:k, 
      U(:,:,i) = trifocal_logarithm(X(:,:,i),Y(:,:,i));
  end

end


 M.hash = @(X) ['z' hashmd5(X(:))];


 M.rand = @() randtrifocal(k);
 function Q = randtrifocal(N)

     if nargin < 1
         N = 1;
     end

     QT = zeros([3 2 N]);
     QT(1:2,:,:) = randn(2,2,N);
     QT =  multiprod(QT,reshape( 1./sqrt( multitrace(multiprod(multitransp(QT),QT))),[1 1 N]));
     Q = [randrot(3,N) randrot(3,N) randrot(3,N) QT ];
 end

 
 M.randvec = @randomvec;
 function U = randomvec(X)
     
     UT = zeros([3 2 k]);
     UT(1:2,:,:) = randn(2,2,k);
     for i=1:k
        UT(:,:,k)  =  UT(:,:,k)  - trace(UT(:,:,k)'*X(:,10:11,k))*X(:,10:11,k);
     end
     
     
     U = [multiprod( p1(X), randskew(3,k))...
          multiprod( p2(X), randskew(3,k))...
          multiprod( p3(X), randskew(3,k))...
          UT];
     
     U = tangentProjection(X,U);
     
     U = U / sqrt(M.inner([],U,U)); 
     
 end

 M.lincomb = @matrixlincomb;

 M.zerovec = @(x) zeros(3, 11, k);

 M.transp = @transport;
 function S2 = transport(X1, X2, S1)
   
    S2 = zeros(3,11,k);
    
    S2(:,1:3,:) = multiprod(X2(:,1:3,:), multiprod(multitransp(X1(:,1:3,:)) , S1(:,1:3,:) ));
    S2(:,4:6,:) = multiprod(X2(:,4:6,:), multiprod(multitransp(X1(:,4:6,:)) , S1(:,4:6,:) ));
    S2(:,7:9,:) = multiprod(X2(:,7:9,:), multiprod(multitransp(X1(:,7:9,:)) , S1(:,7:9,:) ));   
    S2(:,10:11,:) = ParallelTransportSphere(X1(:,10:11,:),X2(:,10:11,:),S1(:,10:11,:)); 
    
 end


 M.pairmean = @pairmean;
 function Y = pairmean(X1, X2)
     V = M.log(X1, X2);
     Y = M.exp(X1, .5*V);
 end

 M.dist = @(x, y) M.norm(x, M.log(x, y)); 
 M.vec = @(x, u_mat) u_mat(:);
 M.mat = @(x, u_vec) reshape(u_vec, [3, 11, k]);
 M.vecmatareisometries = @() true;

 p1 = @(X) X(:,1:3,:);
 p2 = @(X) X(:,4:6,:);
 p3 = @(X) X(:,7:9,:);
 p4 = @(X) X(:,10:11,:);


function V = vertproj(X,H)

    ezhat = [0 -1 0;1 0 0;0 0 0];
    V = zeros(3,11,k);
    
    for i=1:k
        ezhatXi = ezhat*X(:,:,i);
       V(:,:,i) =  .25*ezhatXi *  M.inner(X(:,:,i),H(:,:,i),ezhatXi);
    end
    
end

end


%% Some functions used by the trifocal factory

function v = LogarithmSphere(x1, x2)
        xi = x2-x1;
        v = xi - trace(x1'*xi)*x1;
        di = real(acos(trace(x1'*x2)));
        % If the two points are "far apart", correct the norm.
        if di > 1e-6
            nv = norm(v, 'fro');
            v = v * (di / nv);
        end
end


function Zeta = ParallelTransportSphere(X,Y,Xi,t)

    if nargin <4
        t =1;
    end
    % X,Y: mxnxN where each mxn is a unit Frobenius norm
    
    Zeta =zeros(size(X));
    
    for i=1:size(X,3)
        dx = LogarithmSphere(X(:,:,i),Y(:,:,i));
        ndx = norm(dx,'fro')+eps;
        u = dx/ndx;
        Zeta(:,:,i) = Xi(:,:,i) -X(:,:,i)*sin(ndx*t)*trace(u'*Xi(:,:,i)) -(1-cos(ndx*t))*(u*u')*Xi(:,:,i);
    end

end

function v = vee3(V)
    v = squeeze([V(3,2,:)-V(2,3,:); V(1,3,:)-V(3,1,:); V(2,1,:)-V(1,2,:)])/2;
end

function [V, vShift] = trifocal_hat3(v)
    N = size(v,2);
    V = zeros(3,3,N);
    vShift = permute(v,[1 3 2]);
    V(1,2,:) = -vShift(3,:,:);
    V(2,1,:) = vShift(3,:,:);
    V(1,3,:) = vShift(2,:,:);
    V(3,1,:) = -vShift(2,:,:);
    V(2,3,:) = -vShift(1,:,:);
    V(3,2,:) = vShift(1,:,:);
end

function R = rot3_exp(V)
    v = vee3(V);
    nv = cnorm(v);
    idxZero = nv < 1e-15;
    nvMod = nv;
    nvMod(idxZero) = 1;

    vNorm = v./([1;1;1]*nvMod);

    % Matrix exponential using Rodrigues' formula
    nv = shiftdim(nv,-1);
    c = cos(nv);
    s = sin(nv);
    [VNorm,vNormShift] = trifocal_hat3(vNorm);
    vNormvNormT = multiprod(vNormShift,multitransp(vNormShift));
    R=multiprod(eye(3),c)+multiprod(VNorm,s)+multiprod(vNormvNormT,1-c);
end


function nv = cnorm(v)
	nv = sqrt(sum(v.^2));
end

function y = Sphere_exp(x, d, t)

    if nargin == 2
        t = 1;
    end
    
    td = t*d;
    y = zeros(size(x));
    
    for i=1:size(x,3)
        
        nrm_td = norm(td(:,:,i), 'fro');

        if nrm_td > 1e-6
            y(:,:,i) = x(:,:,i)*cos(nrm_td) + td(:,:,i)*(sin(nrm_td)/nrm_td);
        else
            y(:,:,i) = x(:,:,i) + td(:,:,i);
            y(:,:,i) = y(:,:,i) / norm(y(:,:,i), 'fro');
        end

    end
end
