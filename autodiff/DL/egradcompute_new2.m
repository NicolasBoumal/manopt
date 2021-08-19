function egrad = egradcompute_new2(problem,x)

    persistent tm;
    if isempty(tm)
        tm = deep.internal.recording.TapeManager();
    end
    % tm = deep.internal.recording.TapeManager();
    record = deep.internal.startTracingAndSetupCleanup(tm);
    cost = problem.cost;
    dlx = mat2dl(x);
    dlx2 = deep.internal.recording.recordContainer(dlx);
    [varargout{1}] = cost(dlx2);
    [varargout{2}] = dlgradient(varargout{1},dlx2,'RetainData',true,'EnableHigherDerivatives',true);
    varargout = deep.internal.networkContainerFun(@stopRecording, varargout);
    egrad = dl2mat(varargout{2});
    
end