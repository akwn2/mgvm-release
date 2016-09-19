function series_total = ivd(nu, z, w)
% ivd.m
% Modified bessel function with decomposed argument
% This is a semi-naive implementation (we take care of large arguments
% using logs, but more can be done using log1p)

    tol = 1E-6;
    series_total = 0;
    
    kk = 0;
    aux = log(w.^2 - 1) + log(0.5 .* z);
    log_k_fact = 0;
    
    log_term = log_iv(nu + kk, z) + kk .* aux - log_k_fact;
    term = exp(log_term);
    
    % This series is monotonically decrescent.
    while abs(term) > tol * abs(series_total)
        series_total = series_total + term;
        
        kk = kk + 1;
        log_k_fact = log_k_fact + log(kk);
        log_term = log_iv(nu + kk, z) + kk .* aux - log_k_fact;
        term = exp(log_term);
    end

    series_total = w.^(nu) .* series_total;
end