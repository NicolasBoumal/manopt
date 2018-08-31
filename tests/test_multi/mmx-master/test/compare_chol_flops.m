
%% === compare timings for CHOL

R  = 20;    % number of repeats (to increase accuracy)

nn = 1:1:200;

K  = length(nn);

time   = zeros(K,3,R);
ops    = zeros(K,1);

fun = {
     @(a)mmx('chol',a,[]),...
       @(a)mmx_MMKL('chol',a,[]),...
      @(a)mmx_C('chol',a,[]),...
      @(a)ndfun('chol',a)
       };

names = {
    'mmx Intel',...
    'mmx MKL',...
    'mmx Emo'%,...
%     'ndfun'
    };

for i = 1:K
   
   n = nn(i);

   r1 = n;
   c1 = n;

   N     = round(1e6 / (r1*c1));
   
   A     = randn(n,n,N);
   A     = mmx('s',A,[]);
   AA    = A;
   
   ops(i) = (n^3/3 + n^2/2 +n/6)*N;   

   fprintf('cholesky algorithm on %d SPD matrices of size [%dx%d]\n',N,n,n)

   for r = 1:R
      for f = 1:length(fun)
         tic;
         C = fun{f}(AA);
         time(i,f,r)  = toc;
         AA = A;
      end
   end
end

%%

mflops   = 1e-6*bsxfun(@times, ops, 1./time);
mFLP     = mean(mflops,3);

clf
hold on
cols = get(gca,'ColorOrder');

for i = 1:size(mflops,2)
   plot(nn,mFLP(:,i),'color',cols(i,:),'linewidth',3);
end

for i = 1:size(mflops,2)
   plot(nn,squeeze(mflops(:,i,:)),'.','color',cols(i,:),'linewidth',2)
end

grid on
set(gca,'xlim',[nn(1) nn(end)])
ylabel('Mflops')
xlabel('dimension')

legend(names{:},'location','northwest')

title('\bf Speed Comparison between mmx and ndfun. (in Mflops, bigger is better)')

drawnow;

