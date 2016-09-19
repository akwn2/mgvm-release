function failed = test_inv_log_iv0()
% test_inv_log_iv0
% Tests the log_iv function for calculating the log of a modified bessel
% function of first kind.

    % exponential weighting
    for kk = [0.5, 1, 5, 10, 50, 100, 200, 300, 400, 500, 1000, 1500]
        assert(norm(kk - inv_log_iv0(log(ive(0, kk)), kk)) < 1E-6);
    end

    % fixed exponential reduction
    alpha = 10;
    for kk = [0.5, 1, 5, 10, 50, 100, 200]
        assert(norm(kk -...
            inv_log_iv0(log(iv(0, kk)) - alpha, alpha)) < 1E-6);
    end
    
    failed = 0;
end