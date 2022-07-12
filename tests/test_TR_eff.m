clear; close all; clc;
iters = 4;
r1 = zeros(iters, 1);
t1 = zeros(iters, 1);

r2 = zeros(iters, 1);
t2 = zeros(iters, 1);
l1 = zeros(iters, 1);

l2 = zeros(iters, 1);

n1 = zeros(iters, 1);

n2 = zeros(iters, 1);

for j=1:iters
    n = 3000;
    p = 300;
    m = 150;

        % Data matrix
    A = randn(p, n);

    % Regularization parameter. This should be between 0 and the largest
    % 2-norm of a column of A.
    gamma = 1;
    St = stiefelfactory(p, m);
    problem.M = St;
    
    x = problem.M.rand();
    
    options.subproblemsolver = @trs_tCG_cached;
    options.x = x;
    options.thre = (2*j-2)/(2*iters);
    options.maxiter = 5;
    options.verbosity = 2;
    options2.subproblemsolver = @trs_tCG;
    options2.x = options.x;
    options2.thre = options.thre;
    options2.maxiter = options.maxiter;
    options2.verbosity = options.verbosity;
    if mod(j, 2)
        disp('trs_tCG_cached');
        [~,~,~,~, info] = sparse_pca(A, m, gamma, options);
        disp('trs_tCG');
        [~,~,~,~, infotCG] = sparse_pca(A, m, gamma, options2);
    else 
        disp('trs_tCG');
        [~,~,~,~, infotCG] = sparse_pca(A, m, gamma, options2);
        disp('trs_tCG_cached');
        [~,~,~,~, info] = sparse_pca(A, m, gamma, options);
    end
    assert(isequaln(rmfield(info, 'time'),rmfield(infotCG, 'time')));
    rejCount = 0;
    sumNumInner = 0;
    for i=1:length(info)
        rejCount = rejCount + ~info(i).accepted;
        if i < length(info)
            sumNumInner = sumNumInner + ~info(i).accepted * info(i+1).numinner;
        end
    end
    avgNumInner = sumNumInner/rejCount;
    r1(j) = r1(j) + rejCount;
    l1(j) = l1(j) + length(info);
    n1(j) = n1(j) + avgNumInner;
%     t1(j) = t1(j) + info(1).timeit;
    rejCounttCG = 0;
    sumNumInnertCG = 0;
    for i=1:length(infotCG)
        rejCounttCG = rejCounttCG + ~infotCG(i).accepted;
        if i < length(infotCG)
            sumNumInnertCG = sumNumInnertCG + ~infotCG(i).accepted * infotCG(i+1).numinner;
        end
    end
    avgNumInnertCG = sumNumInnertCG/rejCounttCG;
    
    r2(j) = r2(j) + rejCounttCG;
    l2(j) = l2(j) + length(infotCG);
    n2(j) = n2(j) + avgNumInnertCG;
%     t2(j) =  t2(j) + infotCG(1).timeit;
    fprintf('j = %d\n', j);
    fprintf('time of tCG_cached: %f [s] , tCG: %f [s] \n', t1(j), t2(j));
    fprintf('Proportion of rejections tCG_cached: %d / %d , tCG: %d / %d \n', r1(j), l1(j), r2(j), l2(j));
    fprintf('Average number of iterations for rejection: tCG_cached: %d , tCG: %d \n\n', n1(j), n2(j));
%     elliptope_SDP_complex();
%     shapefit_smoothed();
%     generalized_procrustes();
%     doubly_stochastic_denoising();
%     truncated_svd();
%     essential_svd();
%     dominant_invariant_subspace();
%     dominant_invariant_subspace_complex();
%     low_rank_tensor_completion();
%     low_rank_tensor_completion_embedded();
    close;
end
fprintf('************************************************************************\n');

for i=1:length(r1)
    fprintf('time of tCG_cached: %f [s] , tCG: %f [s] \n', t1(i), t2(i));
    fprintf('Proportion of rejections tCG_cached: %d / %d , tCG: %d / %d \n', r1(i), l1(i), r2(i), l2(i));
    fprintf('Average number of iterations for rejection: tCG_cached: %d , tCG: %d \n\n', n1(i), n2(i));
end

close all;
