mycostfunction = @(x) x'*x;
x = [2;2];
xdot = [2;3];
xdot2 = [4;4];
store = struct();

%tm = deep.internal.recording.TapeManager();
%record = deep.internal.startTracingAndSetupCleanup(tm);

[ehess,store] = ehesscompute_new(mycostfunction,x,xdot,store);
[ehess,store] = ehesscompute_new(mycostfunction,x,xdot2,store)

%z = innerprodgeneral(store.dlegrad,xdot2);
%ehess2 = dlgradient(z,store.dlx,'RetainData',false,'EnableHigherDerivatives',false)



