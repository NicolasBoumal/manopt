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
% Use the function M.tangent2ambient(X, H) to switch from the Lie algebra
% representation to the embedding space representation in R^(3x6).
%
% By default, k = 1.

% This file is part of Manopt: www.manopt.org.
% Original author: Roberto Tron, Aug. 8, 2014
% Contributors: 
    
    
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
    
    e3hat=[0 -1 0; 1 0 0; 0 0 0]; % BM: okay

    M.dim = k*5;% BM: okay
    
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    M.norm = @(x, d) norm(d(:));
    
    M.typicaldist = @() pi*sqrt(2*k);
    
    M.proj = @tangentProjection;
    function HProjHoriz=tangentProjection(X,H)
        %project H on the tangent space of SO(3)^2
        HProj=sharp(multiskew(multiprod(multitransp(flat(X)), flat(H))));
        
        %compute projection on vertical component
        p=M.vertproj(X,HProj);

        HProjHoriz=HProj-multiprod(p/2,[hat3(permute(X(3,1:3,:),[2 3 1])) hat3(permute(X(3,4:6,:),[2 3 1]))]);
    end
    
     
    M.tangent = @(X, H) sharp(multiskew(flat(H)));
    
    M.egrad2rgrad=@egrad2rgrad;
    function rgrad=egrad2rgrad(X,egrad)
        rgrad=M.proj(X,egrad(X));
    end
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, S)
        % Reminder : S contains skew-symmeric matrices. The actual
        % direction that the point X is moved along is X*S.
        RA=M.p1(X);
        RB=M.p2(X);
        SA=M.p1(S);
        SB=M.p2(S);

        dX=M.tangent2ambient(X,S);
        E=M.E(X);
        G=egrad(X);
        GA=M.p1(G);
        GB=M.p2(G);
        
        H=ehess(X,dX);

        %The following is the vectorized version of connection=-[multisym(GA'*RA)*SA multisym(GB'*RB)*SB];
        connection=M.tangent2ambient(X,-cat(2,...
            multiprod(multisym(multiprod(multitransp(GA),RA)),SA),...
            multiprod(multisym(multiprod(multitransp(GB),RB)),SB)));
        
        rhess=M.proj(X,H+connection);
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
    
    M.log = @logarithm;
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
        X12=sharp(multiprod(multitransp(flat(X2)),flat(X1)));
        X12Flat=flat(X12);
        
        %left translation, S2=X12*S*X12'
        S2=sharp(multiprod(X12Flat,multiprod(flat(S1),multitransp(X12Flat))));
    end
    
    M.pairmean = @pairmean;
    function Y = pairmean(X1, X2)
        V = M.log(X1, X2);
        Y = M.exp(X1, .5*V);
    end
    
    M.dist = @(x, y, varargin) M.norm(x, M.log(x, y, varargin{:}));
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, [3, 6, k]);
    M.vecmatareisometries = @() true;
    
    
    %% Robertos functions
    M.p1=@(X) X(:,1:3,:);
    M.p2=@(X) X(:,4:6,:);
    

    M.flat=@flat;
    function Hp=flat(H)
        %Reshape a [3x6xk] matrix to a [3x3x2k] matrix
        Hp=reshape(H,3,3,[]);
    end

    M.sharp=@sharp;
    function H=sharp(Hp)
        %Reshape a [3x3x2k] matrix to a [3x6xk] matrix
        H=reshape(Hp,3,6,[]);
    end

    M.vertproj=@(X,H) multiprod(X(3,1:3,:),permute(vee3(H(:,1:3,:)),[1 3 2]))+multiprod(X(3,4:6,:),permute(vee3(H(:,4:6,:)),[1 3 2]));
    
   M.tangent2ambient = @(X, H) sharp(multiprod(flat(X), flat(H)));
	
    %compute the essential matrix from the quotient representation
    M.E=@(X) multiprod(multiprod(multitransp(M.p1(X)),e3hat),M.p2(X));
    M.dE=@dE;
    function Edot=dE(X,H)
        E=M.E(X);
        Edot=multiprod(multitransp(M.p1(H)),E)+multiprod(E,M.p2(H));
    end

    M.ddE=@ddE;
    function Eddot=ddE(X,S)
        E=M.E(X);
        SA=M.p1(S);
        SB=M.p2(S);
        SASq=multiprod(SA,SA);
        SBSq=multiprod(SB,SB);
        Eddot=multiprod(multitransp(SASq),E)...
            +multiprod(multiprod(multitransp(SA),E),SB)...
            +multiprod(E,SBSq);
    end
    
    M.ef2rf=@(X,ef) ef(M.E(X));

    M.egradE2egrad= @egradE2egrad;
    function egrad=egradE2egrad(X,egradE)
        %Compute the Euclidean gradient with respect to the factors (R1,R2) 
        %starting from the Euclidean gradient with respect to the essential
        %matrix E. In practice, EgradE is a 3x3 matrix, and egrad is a
        %3x3x2 array
        
        RA=M.p1(X);
        RB=M.p2(X);
        E=M.E(X);
        G=egradE(E);
        
        %The following is the vectorized version of egrad=e3hat*[RB*G' -RA*G];
        egrad=multiprod(e3hat,cat(2,...
            multiprod(RB,multitransp(G)),...
            -multiprod(RA,G)));
    end

    

    M.egradE2rgrad= @egradE2rgrad;
    function rgrad=egradE2rgrad(X,egradE)
        %Compute the Riemannian gradient starting from the Euclidean
        %gradient of a function of the 3x3 essential matrix.
        %That is, EgradE is a 3x3 matrix.
        egrad=@(X) M.egradE2egrad(X,egradE);
        rgrad=M.egrad2rgrad(X,egrad);
    end
	
    M.ehessE2ehess=@ehessE2ehess;
    function ehess = ehessE2ehess(X, egradE, ehessE, V)
        % Reminder : V DOES NOT contain skew-symmeric matrices.
        % If you are using this with a tangent vector, you should set V=X*S.

        RA=M.p1(X);
        RB=M.p2(X);
        VA=M.p1(V);
        VB=M.p2(V);

        E=M.E(X);
        %The following is the vectorized version of VA'*e3hat*RB+RA'*e3hat*VB
        dE=multiprod(multitransp(VA),multiprod(e3hat,RB))...
            +multiprod(multitransp(RA),multiprod(e3hat,VB));
        
        G=egradE(E);
        H=ehessE(E,dE);

        %The following is the vectorized version of ehess=e3hat*[(VB*G'+RB*H') -(VA*G+RA*H)]
        ehess=multiprod(e3hat,cat(2,...
            multiprod(VB,multitransp(G))+multiprod(RB,multitransp(H)),...
            -multiprod(VA,G)-multiprod(RA,H)));
	end

	
    
	M.ehessE2rhess = @ehessE2rhess;
    function rhess = ehessE2rhess(X, egradE, ehessE, S)
        egrad=@(X) M.egradE2egrad(X,egradE);
        ehess=@(X,V) M.ehessE2ehess(X, egradE, ehessE, V);
        rhess= M.ehess2rhess(X, egrad, ehess, S);
 	end
    
    
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


