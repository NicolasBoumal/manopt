%function sx=sincN(x)
%Same as sinc, but without the normalization by pi
%
%See also sinc
function sx=sincN(x)
sx=sin(x)./x;
sx(x==0)=1;
