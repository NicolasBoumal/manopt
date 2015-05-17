function M = essentialfactory(k)
% Returns a manifold structure to optimize over the space of essential
% matrices using the quotient representation.
%
% function M = essentialfactory(k)
%
%
% Quotient representation of the essential manifold: deals with the
% representation of the space of essential matrices M_rE. These are used in
% computer vision to represent the epipolar constraint between projected
% points in two perspective views.
%
% The space is represented as the quotient (SO(3)^2/SO(2)). See the
% following references for details
%
%   R. Tron, K. Daniilidis,
%   "On the quotient representation of the essential manifold"
%   IEEE Conference on Computer Vision and Pattern Recognition, 2014
%
% For computational purposes, each essential matrix is represented as a
% [3x6] matrix where each [3x3] block is a rotation.
%
% The metric used is the one induced by the submersion of M_rE in SO(3)^2
%
% Tangent vectors are represented in the Lie algebra of SO(3)^2, i.e., as
% [3x6] matrices where each [3x3] block is a skew-symmetric matrix.
% Use the function tangent2ambient(X, H) to switch from the Lie algebra
% representation to the embedding space representation in R^(3x6).
%
% By default, k = 1.

% This file is part of Manopt: www.manopt.org.
% Original author: Roberto Tron, Aug. 8, 2014
% Contributors: Bamdev Mishra, May 15, 2015.

    
    if ~exist('k', 'var') || isempty(k)
        k = 1; % BM: okay
    end
    
    if k == 1
        M.name = @() sprintf('Quotient representation of the essential manifold'); % BM: okay
    elseif k > 1
        M.name = @() sprintf('Product of %d quotient representations of the essential manifold', k); % BM: okay
    else
        error('k must be an integer no less than 1.'); % BM: okay
    end
    
    M.dim = k*5;% BM: okay
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:); % BM: okay
    
    M.norm = @(x, d) norm(d(:)); % BM: okay
    
    M.typicaldist = @() pi*sqrt(2*k); % BM: okay
    
    M.proj = @tangentProjection; % BM: caution.. Uses sharp and vertproj, but no need to be as structs of M?
    function HProjHoriz=tangentProjection(X,H)
        %project H on the tangent space of SO(3)^2
        HProj = sharp(multiskew(multiprod(multitransp(flat(X)), flat(H)))); % BM: okay
        
        %compute projection on vertical component
        p = vertproj(X, HProj); % BM: okay
        
        HProjHoriz = HProj - multiprod(p/2,[hat3(permute(X(3,1:3,:),[2 3 1])) hat3(permute(X(3,4:6,:),[2 3 1]))]);% BM: okay
    end
    
    
    M.tangent = @(X, H) sharp(multiskew(flat(H))); % BM: okay
    
    M.egrad2rgrad=@egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        % BM
        rgrad = M.proj(X, egrad);
    end
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, S)
        % BM code
        
        % Reminder : S contains skew-symmeric matrices. The actual
        % direction that the point X is moved along is X*S.
        RA=p1(X);
        RB=p2(X);
        SA=p1(S);
        SB=p2(S);
        
        G = egrad; %egrad(X);
        GA = p1(G);
        GB = p2(G);
        
        H = ehess; %ehess(X,dX);
        
        %The following is the vectorized version of connection=-[multisym(GA'*RA)*SA multisym(GB'*RB)*SB];
        connection = tangent2ambient(X,-cat(2,...
            multiprod(multisym(multiprod(multitransp(GA), RA)), SA),...
            multiprod(multisym(multiprod(multitransp(GB), RB)), SB)));
        
        rhess = M.proj(X,H + connection);
    end
    
    
    
    M.exp = @exponential;
    function Y = exponential(X, U, t)
        if nargin == 3
            U = t*U;
        end
        
        UFlat=flat(U);
        exptUFlat=rot3_exp(UFlat);
        Y = sharp(multiprod(flat(X), exptUFlat));
    end
    
    M.retr = @exponential;
    
    M.log = @logarithm; % BM:???
    function U = logarithm(X, Y, varargin)
        flagSigned=true;
        %optional parameters
        ivarargin=1;
        while(ivarargin<=length(varargin))
            switch(lower(varargin{ivarargin}))
                case 'signed'
                    flagSigned=true;
                case 'unsigned'
                    flagSigned=false;
                case 'flagsigned'
                    ivarargin=ivarargin+1;
                    flagSigned=varargin{ivarargin};
                otherwise
                    error(['Argument ' varargin{ivarargin} ' not valid!'])
            end
            ivarargin=ivarargin+1;
        end
        QX=[X(:,1:3,:);X(:,4:6,:)];
        QY=[Y(:,1:3,:);Y(:,4:6,:)];
        QYr=essential_closestRepresentative(QX,QY,'flagSigned',flagSigned);
        Yr=[QYr(1:3,:,:) QYr(4:6,:,:)];
        U=zeros(size(X));
        U(:,1:3,:)=rot3_log(multiprod(multitransp(X(:,1:3,:)),Yr(:,1:3,:)));
        U(:,4:6,:)=rot3_log(multiprod(multitransp(X(:,4:6,:)),Yr(:,4:6,:)));
    end
    
    M.hash = @(X) ['z' hashmd5(X(:))];
    
    M.rand = @() randessential(k);
    
    M.randvec = @randomvec;
    function U = randomvec(X)
        U = tangentProjection(X,sharp(randskew(3, 2*k)));
        U = U / sqrt(M.inner([],U,U));
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(x) zeros(3, 6, k);
    
    M.transp = @transport;
    function S2=transport(X1,X2,S1)
        %transport a vector from the tangent space at X1 to the tangent
        %space at X2. This transport uses the left translation of the
        %ambient group and preserves the norm of S1. The left translation
        %aligns the vertical spaces at the two elements.
        
        %group operation in the ambient group, X12=X2'*X1
        X12 = sharp(multiprod(multitransp(flat(X2)),flat(X1)));
        X12Flat = flat(X12);
        
        %left translation, S2=X12*S*X12'
        S2 = sharp(multiprod(X12Flat,multiprod(flat(S1),multitransp(X12Flat))));
    end
    
    M.pairmean = @pairmean;
    function Y = pairmean(X1, X2)
        V = M.log(X1, X2);
        Y = M.exp(X1, .5*V);
    end
    
    M.dist = @(x, y, varargin) M.norm(x, M.log(x, y, varargin{:})); % BM:???
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, [3, 6, k]);
    M.vecmatareisometries = @() true;
    
    
    
    p1 = @(X) X(:,1:3,:);
    p2 = @(X) X(:,4:6,:);
    
    
    function Hp = flat(H)
        %Reshape a [3x6xk] matrix to a [3x3x2k] matrix
        Hp=reshape(H,3,3,[]);
    end
    
    function H=sharp(Hp)
        %Reshape a [3x3x2k] matrix to a [3x6xk] matrix
        H=reshape(Hp,3,6,[]);
    end
    
    vertproj = @(X,H) multiprod(X(3,1:3,:),permute(vee3(H(:,1:3,:)),[1 3 2]))+multiprod(X(3,4:6,:),permute(vee3(H(:,4:6,:)),[1 3 2])); % BM: okay
    
    tangent2ambient = @(X, H) sharp(multiprod(flat(X), flat(H)));
    
    
end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>
    
    if nargin == 3
        d = a1*d1;
    elseif nargin == 5
        d = a1*d1 + a2*d2;
    else
        error('Bad use of essential.lincomb.');
    end
    
end


%% Some private functions used by the essential factory
function v = vee3(V)
    v = squeeze([V(3,2,:)-V(2,3,:); V(1,3,:)-V(3,1,:); V(2,1,:)-V(1,2,:)])/2;
end


%Compute the exponential map in SO(3) using Rodrigues' formula
% function R=rot3_exp(V)
%V must be a [3x3xN] array of [3x3] skew-symmetric matrices.
function R=rot3_exp(V)
    v=vee3(V);
    nv=cnorm(v);
    idxZero=nv<1e-15;
    nvMod=nv;
    nvMod(idxZero)=1;
    
    vNorm=v./([1;1;1]*nvMod);
    
    %matrix exponential using Rodrigues' formula
    nv=shiftdim(nv,-1);
    c=cos(nv);
    s=sin(nv);
    [VNorm,vNormShift]=hat3(vNorm);
    vNormvNormT=multiprod(vNormShift,multitransp(vNormShift));
    R=multiprod(eye(3),c)+multiprod(VNorm,s)+multiprod(vNormvNormT,1-c);
end

%Compute the matrix representation of the cross product
%function [V,vShift]=hat3(v)
%V is a [3x3xN] array of skew-symmetric matrices where each [3x3] block is
%the matrix representation of the cross product of one of the columns of v
%vShift is equal to permute(v,[1 3 2]).
function [V,vShift]=hat3(v)
    N=size(v,2);
    V=zeros(3,3,N);
    vShift=permute(v,[1 3 2]);
    V(1,2,:)=-vShift(3,:,:);
    V(2,1,:)=vShift(3,:,:);
    V(1,3,:)=vShift(2,:,:);
    V(3,1,:)=-vShift(2,:,:);
    V(2,3,:)=-vShift(1,:,:);
    V(3,2,:)=vShift(1,:,:);
end

%Compute the logarithm map in SO(3)
% function V=rot3_log(R)
%V is a [3x3xN] array of [3x3] skew-symmetric matrices
function V=rot3_log(R)
    skewR=multiskew(R);
    ctheta=(multitrace(R)'-1)/2;
    stheta=cnorm(vee3(skewR));
    theta=atan2(stheta,ctheta);
    
    V=skewR;
    for ik=1:size(R,3)
        V(:,:,ik)=V(:,:,ik)/sincN(theta(ik));
    end
end

function Q = randessential(N)
% Generates random essential matrices.
%
% function Q = randessential(N)
%
% Q is a [3x6] matrix where each [3x3] block is a uniformly distributed
% matrix.

% This file is part of Manopt: www.manopt.org.
% Original author: Roberto Tron, Aug. 8, 2014
% Contributors:
% Change log:
    
    if nargin < 1
        N = 1;
    end
    
    Q = [randrot(3,N) randrot(3,N)];
    
end



function sx = sincN(x)
    sx = sin(x)./x;
    sx(x==0) = 1;
end

function nv = cnorm(v)
    nv = sqrt(sum(v.^2));
end

function Q2r=essential_closestRepresentative(Q1,Q2,varargin)
    [tMin,~,Q2]=essential_distMinAngle(Q1,Q2,varargin{:});
    NQ1=size(Q1,3);
    NQ2=size(Q2,3);
    
    if NQ1>1 && NQ2==1
        Q2=repmat(Q2,[1 1 NQ1]);
    end
    NQ=max(NQ1,NQ2);
    
    Q2r=zeros(size(Q2));
    for iQ=1:NQ
        t=tMin(iQ);
        Rz=[cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1];
        Q2r(1:3,1:3,iQ)=Rz*Q2(1:3,1:3,iQ);
        Q2r(4:6,1:3,iQ)=Rz*Q2(4:6,1:3,iQ);
    end
    
end



function [tMin,fMin,Q2Flip,output]=essential_distMinAngle(Q1,Q2,varargin)
    NQ1=size(Q1,3);
    NQ2=size(Q2,3);
    
    if NQ1==1 && NQ2>1
        Q1=repmat(Q1,[1 1 NQ2]);
        NQ1=NQ2;
    end
    if NQ1>1 && NQ2==1
        Q2=repmat(Q2,[1 1 NQ1]);
    end
    
    if NQ1>1
        tMin=zeros(NQ1,1);
        fMin=zeros(NQ1,1);
        Q2Flip=zeros(6,3,NQ1);
        if nargout>3
            output=repmat(struct('tMin',[],'fMin',[],'tBreak1',[],'tBreak2',[]),NQ1,1);
        end
        for iQ=1:NQ1
            if nargout>3
                [tMin(iQ),fMin(iQ),Q2Flip(:,:,iQ),output(iQ)]=...
                    essential_distMinAngle(Q1(:,:,iQ),Q2(:,:,iQ),varargin{:});
            else
                [tMin(iQ),fMin(iQ),Q2Flip(:,:,iQ)]=...
                    essential_distMinAngle(Q1(:,:,iQ),Q2(:,:,iQ),varargin{:});
            end
        end
    else
        flagModTMin=false;
        flagSigned=false;
        
        %optional parameters
        ivarargin=1;
        while(ivarargin<=length(varargin))
            switch(lower(varargin{ivarargin}))
                case 'flagmodtmin'
                    ivarargin=ivarargin+1;
                    flagModTMin=varargin{ivarargin};
                case 'signed'
                    flagSigned=true;
                case 'flagsigned'
                    ivarargin=ivarargin+1;
                    flagSigned=varargin{ivarargin};
                otherwise
                    error(['Argument ' varargin{ivarargin} ' not valid!'])
            end
            ivarargin=ivarargin+1;
        end
        
        tMin=zeros(4,1);
        fMin=zeros(4,1);
        tBreak1=zeros(4,1);
        tBreak2=zeros(4,1);
        Q2Flip=zeros(6,3,4);
        if ~flagSigned
            for k=1:4
                [tMin(k),fMin(k),tBreak1(k),tBreak2(k),Q2Flip(:,:,k)]=...
                    essential_distMinAnglePair(Q1,Q2,k);
            end
        else
            [tMin,fMin,tBreak1,tBreak2,Q2Flip]=...
                essential_distMinAnglePair(Q1,Q2,1);
        end
        
        if flagModTMin
            tMin=modAngle(tMin);
        end
        
        if nargout>3
            output.tMin=tMin;
            output.fMin=fMin;
            output.tBreak1=tBreak1;
            output.tBreak2=tBreak2;
        end
        
        if ~flagSigned
            [fMin,idxMin]=min(fMin);
            fMin=max(fMin,0);
            tMin=tMin(idxMin);
            Q2Flip=Q2Flip(:,:,idxMin);
            if nargout>3
                output.idxMin=idxMin;
            end
        end
    end
end


function [tMin,fMin,tBreak1,tBreak2,Q2,tMinAll]=essential_distMinAnglePair(Q1,Q2,kFlip)
    
    switch kFlip
        case 1
            %nothing to do
        case 2
            Q2([2 3 4 6],:)=-Q2([2 3 4 6],:);
        case 3
            Q2([4 5],:)=-Q2([4 5],:);
        case 4
            Q2([2 3 5 6],:)=-Q2([2 3 5 6],:);
        otherwise
            error('Value of kFlip invalid')
    end
    
    Q11=Q1(1:3,:);
    Q12=Q1(4:6,:);
    Q21=Q2(1:3,:);
    Q22=Q2(4:6,:);
    
    Q211=Q21*Q11';
    Q212=Q22*Q12';
    [tMin,fMin,tBreak1,tBreak2,tMinAll]=essential_distMinAnglePair_base(Q211,Q212);
    
end



function [tMin,fMin,tBreak1,tBreak2,tMinAll]=essential_distMinAnglePair_base(Q211,Q212)
    flagCheckFirstDer=true;
    flagUseNewton=true;     %Note: requires flagCheckFirstDer=true
    tolMZero=1e-15;
    tMinAll=[];
    
    [tBreak1,~,~,c1,m1,p1]=essential_distMinAnglePair_discontinuityDistance(Q211);
    [tBreak2,~,~,c2,m2,p2]=essential_distMinAnglePair_discontinuityDistance(Q212);
    
    %check for the degenerate case where the cost is constant
    if abs(m1)<tolMZero && abs(m2)<tolMZero
        tMin=0;
        fMin=2*pi^2;
        tMinAll=0;
    else
        %ft=@(t)  acos((m1*sin(t+p1)+c1-1)/2)^2+acos((m2*sin(t+p2)+c2-1)/2)^2;
        
        if abs(modAngle(tBreak1-tBreak2))<1e-8
            tMin=tBreak1+pi;
            fMin=0;
            %         theta1=@(t) acos((m1*sin(t+p1)+c1-1)/2);
            %         theta2=@(t) acos((m2*sin(t+p2)+c2-1)/2);
            %
            %         ft=@(t) 0.5*(theta1(t)^2+theta2(t)^2);
            %         [tMin,fMin]=fminbnd(ft,tBreak1,tBreak1+2*pi);
        else
            tSearch1=tBreak1;
            tSearch2=tBreak2;
            if tSearch1>tSearch2
                tSearch1=tSearch1-2*pi;
            end
            
            if flagCheckFirstDer
                %compute derivatives of each term at discontinuity points
                df1Break1=essential_distMinAnglePair_computeDfBreak(tBreak1,Q211);
                df2Break2=essential_distMinAnglePair_computeDfBreak(tBreak2,Q212);
                %             disp('[df1Break1 df2Break2]')
                %             disp([df1Break1 df2Break2])
                %compute derivative of each term at other's discontinuity
                %(unroll two calls to dfi)
                theta1Break2=acos(clip((m1*sin(tBreak2+p1)+c1-1)/2));
                df1Break2=-theta1Break2*(m1*cos(tBreak2+p1))/(2*sin(theta1Break2));
                theta2Break1=acos(clip((m2*sin(tBreak1+p2)+c2-1)/2));
                df2Break1=-theta2Break1*(m2*cos(tBreak1+p2))/(2*sin(theta2Break1));
                
                %compute left and right derivatives of sum of the two terms
                dfBreak1n=+df1Break1+df2Break1;
                dfBreak1p=-df1Break1+df2Break1;
                dfBreak2n=+df2Break2+df1Break2;
                dfBreak2p=-df2Break2+df1Break2;
                
                flagSearch1=false;
                %     plot([tBreak1 tBreak2],[dfBreak1p dfBreak2p],'cx','MarkerSize',10)
                %     plot([tBreak1 tBreak2],[dfBreak1n dfBreak2n],'mx','MarkerSize',10)
                if sign(dfBreak1p)~=sign(dfBreak2n)
                    if flagUseNewton
                        %parabolic prediction of min
                        tMin0=tSearch1-dfBreak1p*(tSearch2-tSearch1)/(dfBreak2n-dfBreak1p);
                        %tMin0=(tSearch1+tSearch2)/2;
                        [tMin,fMin]=essential_distMinAnglePair_dfNewton(m1,p1,c1,m2,p2,c2,tMin0,tSearch1,tSearch2);
                        %fMin=essential_distMinAnglePair_ft(m1,p1,c1,m2,p2,c2,tMin);
                    else
                        [tMin,fMin]=fminbnd(essential_distMinAnglePair_ft,tSearch1,tSearch2);
                    end
                    tMinAll=[tMinAll tMin];
                    flagSearch1=true;
                end
                tSearch1=tSearch1+2*pi;
                if sign(dfBreak2p)~=sign(dfBreak1n)
                    if flagUseNewton
                        %parabolic prediction of min
                        tMin0=tSearch2-dfBreak2p*(tSearch1-tSearch2)/(dfBreak1n-dfBreak2p);
                        %tMin0=(tSearch1+tSearch2)/2;
                        [tMin2,fMin2]=essential_distMinAnglePair_dfNewton(m1,p1,c1,m2,p2,c2,tMin0,tSearch2,tSearch1);
                        %fMin2=essential_distMinAnglePair_ft(m1,p1,c1,m2,p2,c2,tMin2);
                    else
                        [tMin2,fMin2]=fminbnd(essential_distMinAnglePair_ft,tSearch2,tSearch1);
                    end
                    if ~flagSearch1 || (flagSearch1 && fMin2<fMin)
                        tMin=tMin2;
                        fMin=fMin2;
                    end
                    tMinAll=[tMinAll tMin2];
                end
            else
                [tMin1,fMin1]=fminbnd(essential_distMinAnglePair_ft,tSearch1,tSearch2);
                tSearch1=tSearch1+2*pi;
                [tMin2,fMin2]=fminbnd(essential_distMinAnglePair_ft,tSearch2,tSearch1);
                if fMin1<fMin2
                    tMin=tMin1;
                    fMin=fMin1;
                else
                    tMin=tMin2;
                    fMin=fMin2;
                end
            end
        end
    end
    
    function v=clip(v)
        v=min(1,max(-1,v));
        
        
        % function f=fi(m,p,c,t)
        % f=acos((m*sin(t+p)+c-1)/2);
        %
        % function d=dfi2(m,p,theta,t)
        % dtheta= -(m*cos(t+p))/(2*sin(theta));
        % d=theta*dtheta;
        %
        % function dd=ddfi2(m,p,theta,t)
        % eztuSq=(m*cos(t+p)/(2*sin(theta)))^2;
        % dd=eztuSq+theta/2*cot(theta/2)*(1-eztuSq);
        %
        % function d=dfi(m,p,c,t)
        % theta=acos((m*sin(t+p)+c-1)/2);
        % dtheta= -(m*cos(t+p))/(2*sin(theta));
        % d=theta*dtheta;
        %
        % function dd=ddfi(m,p,c,t)
        % theta=acos((m*sin(t+p)+c-1)/2);
        % eztuSq=(m*cos(t+p)/(2*sin(theta)))^2;
        % dd=eztuSq+theta/2*cot(theta/2)*(1-eztuSq);
        
        
        
    end
    
end

function dfBreak=essential_distMinAnglePair_computeDfBreak(tBreak,Q21)
    c=cos(tBreak);
    s=sin(tBreak);
    
    % The code below is an optimization exploiting the structure of RBreak to
    % substitute the following code
    %     RBreak=Q1'*[c -s 0; s c 0; 0 0 1]*Q2;
    %
    %     %compute v0 such that RBreak=rot(pi*v0)
    %     [U,~,~]=svd(RBreak+eye(3));
    %     v0=U(:,1);
    %
    %     dfBreak=pi*abs(Q1(3,:)*v0);
    
    Q1RBreakQ1=[c -s 0; s c 0; 0 0 1]*Q21;
    [U,~,~]=svd(Q1RBreakQ1+eye(3));
    dfBreak=pi*abs(U(3,1));
end


%Support function for essential_distMinAnglePair implementing Newton's search
function [tMin,fMin]=essential_distMinAnglePair_dfNewton(m1,p1,c1,m2,p2,c2,tMin,tLow,tHigh)
    tolDist=1e-8;
    for i=1:100
        %     d=dfi(m1,p1,c1,tMin)+dfi(m2,p2,c2,tMin);
        %     dd=ddfi(m1,p1,c1,tMin)+ddfi(m2,p2,c2,tMin);
        %The code below unrolls the following calls
        %     f1=fi(m1,p1,c1,tMin);
        %     f2=fi(m2,p2,c2,tMin);
        %     d=dfi2(m1,p1,f1,tMin)+dfi2(m2,p2,f2,tMin);
        %     dd=ddfi2(m1,p1,f1,tMin)+ddfi2(m2,p2,f2,tMin);
        mc1=m1*cos(tMin+p1);
        mc2=m2*cos(tMin+p2);
        f1=acos(clip((m1*sin(tMin+p1)+c1-1)/2));
        f2=acos(clip((m2*sin(tMin+p2)+c2-1)/2));
        sf1=2*sin(f1);
        sf2=2*sin(f2);
        d1=-f1*mc1/sf1;
        d2=-f2*mc2/sf2;
        d=d1+d2;
        eztuSq1=(mc1/sf1)^2;
        dd1=eztuSq1+f1/2*cot(f1/2)*(1-eztuSq1);
        eztuSq2=(mc2/sf2)^2;
        dd2=eztuSq2+f2/2*cot(f2/2)*(1-eztuSq2);
        dd=dd1+dd2;
        
        
        tOld=tMin;
        tMin=max(tLow+tolDist,min(tHigh-tolDist,tOld-d/dd));
        if abs(tMin-tOld)<tolDist
            break
        end
    end
    fMin=f1^2+f2^2;
    
    function v=clip(v)
        v=min(1,max(-1,v));
        
    end
end



function [tBreak,a,b,c,m,p]=essential_distMinAnglePair_discontinuityDistance(Q21)
    a=Q21(1,1)+Q21(2,2);
    b=Q21(1,2)-Q21(2,1);
    c=Q21(3,3);
    
    m=norm([a;b]);
    p=sign(a)*acos(clip(b/m));
    
    %tBreak=modAngle(3/2*pi-p);
    tBreak=-0.5*pi-p;
    
    function v=clip(v)
        v=min(1,max(-1,v));
    end
end



%Evaluate cost function for closest representative search given coefficients
%function ft=essential_distMinAnglePair_ft(t,m1,p1,c1,m2,p2,c2)
%Evaluates the cost function used by essential_distMinAnglePair to find the
%closest representative in the equivalence class of a QREM
%If m2,p2,c2 are omitted or empty, get value of a single term
function ft=essential_distMinAnglePair_ft(t,m1,p1,c1,m2,p2,c2)
    flagSingleTerm=false;
    if ~exist('m2','var') || isempty(m2)
        flagSingleTerm=true;
    end
    
    if flagSingleTerm
        ft=acos((m1*sin(t+p1)+c1-1)/2)^2;
    else
        ft=acos((m1*sin(t+p1)+c1-1)/2)^2+acos((m2*sin(t+p2)+c2-1)/2)^2;
    end
end

function [ft,tBreak]=essential_distMinAnglePair_ftFromQ(t,Q1,Q2,varargin)
    kFlip=1;
    term='both';
    
    ivarargin=1;
    while(ivarargin<=length(varargin))
        switch(lower(varargin{ivarargin}))
            case 'kflip'
                ivarargin=ivarargin+1;
                kFlip=varargin{ivarargin};
            case 'term'
                ivarargin=ivarargin+1;
                term=lower(varargin{ivarargin});
            otherwise
                disp(varargin{ivarargin})
                error('Argument not valid!')
        end
        ivarargin=ivarargin+1;
    end
    
    
    Q2=essential_flipAmbiguity(Q2,kFlip);
    
    tBreak=[];
    ft=0;
    if strcmp(term,'first') || strcmp(term,'both')
        Q11=essential_getR1(Q1);
        Q21=essential_getR1(Q2);
        Q211=Q21*Q11';
        [tBreak1,~,~,c1,m1,p1]=essential_distMinAnglePair_discontinuityDistance(Q211);
        tBreak=[tBreak tBreak1];
        ft=ft+essential_distMinAnglePair_ft(t,m1,p1,c1);
    end
    
    if strcmp(term,'second') || strcmp(term,'both')
        Q22=essential_getR2(Q2);
        Q12=essential_getR2(Q1);
        Q212=Q22*Q12';
        [tBreak2,~,~,c2,m2,p2]=essential_distMinAnglePair_discontinuityDistance(Q212);
        tBreak=[tBreak tBreak2];
        ft=ft+essential_distMinAnglePair_ft(t,m2,p2,c2);
    end
end


