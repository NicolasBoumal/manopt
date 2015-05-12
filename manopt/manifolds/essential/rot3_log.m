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
