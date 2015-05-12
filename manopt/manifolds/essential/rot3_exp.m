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
