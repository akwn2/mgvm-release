function failed = test_log_iv()
% test_log_iv
% Tests the log_iv function for calculating the log of a modified bessel
% function of first kind.
    
    for kk = [0.5, 1, 5, 10, 50, 100, 200]
        for nn = 0:4
            assert(norm(log(iv(nn, kk)) - log_iv(nn, kk)) < 1E-6);
        end
    end

    failed = 0;
end