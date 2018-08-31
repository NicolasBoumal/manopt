function c = matlab_mprod(a,b)
c  = zeros(size(a,1),size(b,2),size(a,3));
for i = 1:size(a,3)
   c(:,:,i) = a(:,:,i)*b(:,:,i);
end