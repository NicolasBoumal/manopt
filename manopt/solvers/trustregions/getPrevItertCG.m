function [e_Pe, e_Pd, d_Pd, d_Hd, mdelta, Heta, inner_it] = getPrevItertCG(...
                                        Delta, storedb, key)
    
    store = storedb.getWithShared(key);
    store_iters = store.store_iters;

    for i=1:length(store_iters)
        normsq = store_iters(i).normsq;
        d_Hd = store_iters(i).d_Hd;
        if isempty(store_iters(i).normsq) % This means we exit using 
            
        end
        if d_Hd <= 0 || normsq >= Delta^2
            disp(d_Hd);
            e_Pe = store_iters(i).e_Pe;
            e_Pd = store_iters(i).e_Pd;
            d_Pd = store_iters(i).d_Pd;
            mdelta = store_iters(i).mdelta;
            Heta = store_iters(i).Heta;
            inner_it = store_iters(i).inner_it;
            break;
        end
    end
end