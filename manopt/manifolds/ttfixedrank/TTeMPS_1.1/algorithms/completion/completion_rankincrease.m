function [X,cost,test,stats] = completion_rankincrease_adaptive_new2( method, A_Omega, Omega, A_Omega_C, Omega_C, A_Gamma, Gamma, X0, opts )

    if ~isfield( opts, 'maxrank');      opts.maxrank = 4  ;         end
    if ~isfield( opts, 'cg');           opts.cg = true;             end
    if ~isfield( opts, 'tol');          opts.tol = 1e-6;            end
    if ~isfield( opts, 'reltol');       opts.reltol = 1e-8;         end
    if ~isfield( opts, 'reltol_final'); opts.reltol_final = eps;    end
    if ~isfield( opts, 'maxiter');      opts.maxiter = 10;          end
    if ~isfield( opts, 'maxiter_final');opts.maxiter_final = 20;    end
    if ~isfield( opts, 'locked_tol');   opts.locked_tol = 1;        end
    if ~isfield( opts, 'epsilon');      opts.epsilon = 1e-8;        end
    if ~isfield( opts, 'verbose');      opts.verbose = false;       end

    if strcmpi( method, 'GeomCG' )
        completion = @( A_Omega, Omega, A_Gamma, Gamma, X0, opts ) ...
                            completion_orth( A_Omega, Omega, A_Gamma, Gamma, X0, opts )
    elseif strcmpi( method, 'ALS' )
        completion = @( A_Omega, Omega, A_Gamma, Gamma, X0, opts ) ...
                            completion_als( A_Omega, Omega, A_Gamma, Gamma, X0, opts )
    end
    d = X0.order;

    test = [];
    control_old = inf;

    % ===========================================
    disp('____________________________________________________________________');
    disp(['Completion with with starting rank r = [ ' num2str(X0.rank) ' ] ...']);
    [X,cost,control,stats] = completion( A_Omega, Omega, A_Gamma, Gamma, X0, opts);

    stats.rankidx = [length(cost)];

    disp('____________________________________________________________________');
    disp(['Increasing rank ... ']);

    locked = zeros(1,d+1);

    for k = 2:opts.maxrank
        for i = 2:d
        
            disp(['Locked cores:' num2str(locked) ])
            if locked(i)
                disp(['Rank r(' num2str(i) ') is locked. Skipping.']);
            else
                r = X.rank;
                disp(['Trying to increase rank r(' num2str(i) ') from ' num2str(r(i)) ' to ' num2str(r(i)+1) ':']);
                Xnew = increaseRank(X, 1, i, opts.epsilon);
                Xnew = orthogonalize(Xnew, d);
                if i==d && k == opts.maxrank 
                    opts.maxiter = opts.maxiter_final;
                end
                [Xnew,cost_tmp,control_tmp,stats_tmp] = completion( A_Omega, Omega, A_Omega_C, Omega_C, Xnew, opts);
                stats.rankidx = [stats.rankidx, length(cost_tmp)];
                disp( ['Current cost function:            ', num2str(cost_tmp(end)) ]);

                progress = (control_tmp(end) - control_old )/control_old;
                disp( ['Current rel. progress on control: ' num2str(progress)]);

                if  progress > opts.locked_tol
                    disp(['     ... failed. Reverting.']);
                else
                   disp(['     ... accepted.']);
                   X = Xnew;
                   control_old = control_tmp(end)
                   test_current = norm(X(Gamma) - A_Gamma)/ norm(A_Gamma)
                   disp( ['Current error on test set Gamma:  ', num2str(test_current) ]);
                   test = [test, test_current];
                end
            
                if ~isempty(stats.time)
                    stats_tmp.time = stats_tmp.time + stats.time(end);
                end
                
                cost = [cost; cost_tmp];
                control = [control; control_tmp];
                stats.time = [stats.time, stats_tmp.time];
                
            end

        end
    end

end
