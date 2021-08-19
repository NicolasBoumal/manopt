function x = stopRecording(x)
    if isa(x, 'dlarray')
        x = stop(x);
    end
end