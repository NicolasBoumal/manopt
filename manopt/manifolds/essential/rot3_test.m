function rot3_exp_test
V=randn(3,3,2);
V=V-multitransp(V);

R=rot3_exp(V);
disp('Check that rot3_exp gives rotation matrices')
for n=1:size(R,3)
    disp(R(:,:,n)'*R(:,:,n))
    disp(det(R(:,:,n)))
end

disp('Check that rot3_log gives skew-symmetric matrices')
V1=rot3_log(R);
disp(V1-multiskew(V1))

disp('Check that rot3_exp and rot3_log are inverse of each other')
disp(V-V1)
