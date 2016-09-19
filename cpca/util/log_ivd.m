function y = log_ivd(nu, z, w)
% log_ivd.m
% Modified bessel function with decomposed argument
    
    tol = 1E-6;
    log_total = 1;
    
    kk = 0;
    aux = log(exp(2 .* w) - 1) + log(0.5 .* z);
    log_k_fact = 0;
    
    log_term = log_iv(nu + kk, z) + kk .* aux - log_k_fact;
    
    % This series is monotonically decrescent.
    while log_term > tol * log_total
        log_total = log_total + log1p( exp(log_term - log_total));
        
        kk = kk + 1;
        log_k_fact = log_k_fact + log(kk);
        log_term = log_iv(nu + kk, z) + kk .* aux - log_k_fact;
    end
    
    y = log_total + nu .* w;
end