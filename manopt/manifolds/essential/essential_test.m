function essential_test

k=2;
M=essentialfactory(k);
X=randessential(k);

disp('Check that X contains rotation matrices')
testRotations(X)

H=M.proj(X,randn(3,6,k));
disp('Check that projection in tangent space gives skew symmetric matrices')
for ik=1:k
    H1=H(:,1:3,ik);
    H2=H(:,4:6,ik);
    disp([H1+H1' H2+H2'])
end

disp('Check that projection in tangent space gives horizontal vectors')
disp(squeeze(M.vertproj(X,H)))

disp('Check that projection of a tangent vector in ambient space is itself')
testTangents(X,H)

X1=M.exp(X,H);

disp('Check that output of exponential contains rotation matrices')
testRotations(X1)

H1=M.log(X,X1);
disp('Check that log is the inverse of exp')
disp(H-H1);

H1b=M.log(X,X1,'signed');
disp('Check that log and distance are, by default, on the signed manifold')
disp(H-H1b);

disp('Check that the transport operator gives tangent vectors')
X1=M.rand();
X2=M.rand();
H1=M.randvec(X1);
H2=M.transp(X1,X2,H1);
testTangents(X2,H2);


function testRotations(X)
    for ik=1:k
        XA=X(:,1:3,ik);
        XB=X(:,4:6,ik);
        disp([XA'*XA XB'*XB])
        disp([det(XA) det(XB)]);
    end
end

function testTangents(X,H)
    HAmbient=M.tangent2ambient(X,H);
    HReproj=M.proj(X,HAmbient);
    for ik=1:k
        disp(H(:,:,ik)-HReproj(:,:,ik))
    end
end

end
