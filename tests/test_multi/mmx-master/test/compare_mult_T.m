%% === compare timings for MULT

% number of repeats (to increase measurement accuracy)
R  = 10;        

% what sizes of (square) matrices do we want measure? (log spacing)
nn = unique(round(exp(linspace(log(4),log(100),10))));  

% number of different sizes to try
N  = length(nn);

% size of the output C, in Megabytes
Mbytes = 10;

time   = zeros(N,3,R);
ops    = zeros(N,1);

candidates = {'A*B',  'mmx_naive',  @(a,b)mmx_naive('m',a,b);
              'A''*B',  'mmx_naive',  @(a,b)mmx_naive('m',a,b,'tn');
              'A*B''',  'mmx_naive',  @(a,b)mmx_naive('m',a,b,'nt');
              'A''*B''',  'mmx_naive',  @(a,b)mmx_naive('m',a,b,'tt');
              'A*Bs',  'mmx_naive',  @(a,b)mmx_naive('m',a,b,'ns');
              'As*B',  'mmx_naive',  @(a,b)mmx_naive('m',a,b,'sn');
};

funcs = cell(0,0);
for i=1:size(candidates,1)
   found_it = ~isempty(which(candidates{i,2}));
   if found_it
      funcs(end+1,:) = candidates(i,[1 3]); %#ok<SAGROW>
   else
      fprintf('Could not find %s in your path, ignoring.\n',candidates{i,2});
   end
end
nf    = length(funcs);

for i = 1:N
    
   n = nn(i);

   r1 = n;
   c1 = n;
   r2 = c1;
   c2 = n;

   pages = round(1e6*Mbytes / (8*r1*c2));
   
   A     = rand(r1,c1,pages);
   B     = rand(r2,c2,pages);   
   
   B(B>0.5) = 0;
   A(A>0.5) = 0;
   
   ops(i)   = r1*c2*(2*c1-1)*pages;   
   fprintf('multiplying %d*%d matrix pairs of dimension [%dx%d]\n',R*nf,pages,n,n)

   for r = 1:R
      for f = 1:nf %nf:-1:1
         tstart = tic;
         C = funcs{length(funcs)+1-f,2}(A,B);
         time(i,length(funcs)+1-f,r)  = toc(tstart);
         pause(1/10000);
      end
   end
end

%% graphics

gflops   = 1e-9*bsxfun(@times, ops, 1./time);
gFLP     = mean(gflops,3);

clf
hold on 
cols = get(gca,'ColorOrder');
cols = [cols; cols];
for i = 1:size(gflops,2)
   plot(nn,gFLP(:,i),'color',cols(i,:),'linewidth',3);
end
for i = 1:size(gflops,2)
   plot(nn,squeeze(gflops(:,i,:)),'.','color',cols(i,:))
end

grid on
set(gca,'xlim',[1 max(nn)])
ylabel('\bf Gigaflops')
xlabel('\bf dimension')

legend(funcs{:,1},'location','northwest')

title('\bf Comparison between mmx, ndfun, and mtimesx. (in Gflops, bigger is better)')

%% make PDF
% set(gcf,'color','w')
% export_fig 'comparison' -png