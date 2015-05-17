
function Test_essential_svd
%Sample solution of an optimization problem on the essential manifold
%Solves the problem \sum_{i=1}^N ||E_i-A_i||^2, where E_i are essential
%matrices.
%Make data for the test
N = 2;    %number of matrices to process in parallel
A = multiprod(multiprod(randrot(3, N), hat3([0; 0; 1])), randrot(3, N));

%Make the manifold
M = essentialfactory(N);
problem.M = M;


problem.cost = @cost;
    function val = cost(X)
        e3hat = [0 -1 0; 1 0 0; 0 0 0];
        RA = X(:,1:3,:); % M.p1(X);
        RB = X(:,4:6,:); % M.p2(X);
        E = multiprod(multiprod(multitransp(RA), e3hat), RB); % M.E(X);
        G =  E - A;
        val = 0.5*sum(multitrace(multiprod(multitransp(G),(G))));
    end

problem.egrad = @egrad;

    function g = egrad(X)
        e3hat = [0 -1 0; 1 0 0; 0 0 0];
        RA = X(:,1:3,:); % M.p1(X);
        RB = X(:,4:6,:); % M.p2(X);
        E = multiprod(multiprod(multitransp(RA), e3hat), RB); % M.E(X);
        G =  E - A;
        
        %The following is the vectorized version of egrad=e3hat*[RB*G' -RA*G];
        g = multiprod(e3hat, cat(2,...
            multiprod(RB, multitransp(G)),...
            -multiprod(RA, G)));
    end

problem.ehess = @ehess;
    function gdot = ehess(X, S)
        e3hat = [0 -1 0; 1 0 0; 0 0 0];
        
        RA = X(:,1:3,:); % M.p1(X);
        RB = X(:,4:6,:); % M.p2(X);
        E = multiprod(multiprod(multitransp(RA), e3hat), RB); % M.E(X);
        G =  E - A;
       
        V = sharp(multiprod(flat(X), flat(S)));
        VA = V(:,1:3,:); % M.p1(V);
        VB = V(:,4:6,:); % M.p2(V);
        
        dE = multiprod(multiprod(multitransp(RA), e3hat), VB)...
            + multiprod(multiprod(multitransp(VA), e3hat), RB); 
        dG = dE; 
        
        %The following is the vectorized version of ehess=e3hat*[(VB*G'+RB*H') -(VA*G+RA*H)]
        gdot = multiprod(e3hat,cat(2,...
            multiprod(VB, multitransp(G)) + multiprod(RB, multitransp(dG)),...
            -multiprod(VA, G) - multiprod(RA, dG)));
        
    end
    
    
    
    % Numerically check the differentials.
    checkgradient(problem); pause;
    checkhessian(problem); pause;
    
    
    
    %Solve the problem
    X = trustregions(problem);
end

%Compute the matrix representation of the cross product
%function [V,vShift]=hat3(v)
%V is a [3x3xN] array of skew-symmetric matrices where each [3x3] block is
%the matrix representation of the cross product of one of the columns of v
%vShift is equal to permute(v,[1 3 2]).
function [V, vShift] = hat3(v)
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

function H = sharp(Hp)
        %Reshape a [3x3x2k] matrix to a [3x6xk] matrix
        H = reshape(Hp,3,6,[]);
end
    
function Hp = flat(H)
    %Reshape a [3x6xk] matrix to a [3x3x2k] matrix
    Hp=reshape(H,3,3,[]);
end

