function T = robust_gvm_series(r, m1, m2, k1, k2)
% robust_gvm_series
% calculates the rth moment of the Generalised von Mises distribution
% by series approximations in complex number form.

    [R, C] = size(m1);
    tol = 1E-6;
    m1 = m1(:);
    m2 = m2(:);
    k1 = k1(:);
    k2 = k2(:);

    delta = m2 - m1;
    
    % Calculate the r-th trigonometric moment
    varphi_r = ive(0, k2(:, 1)) .* ive(r, k1(:, 1));
    
%     w1 = log(k1);
%     k1_hat = k1 .* exp(-w1);
%     w2 = k2 ./ k1_hat;
    
    jj = 0;
    term = 0;
    store_a = []
    store_b = []
    a = 0;
    b = varphi_r;
    while (a + b) > tol * abs(varphi_r)

        jj = jj + 1;
% 
%         a = exp(log_ivd(jj, k1_hat, w2) + ...
%                 log_ivd(2 .* jj + r, k1_hat, w1));
%             
%         b = exp(log_ivd(jj, k1_hat, w2) + ...
%                 log_ivd(abs(2 .* jj - r), k1_hat, w1));

        a = exp(log_iv(jj, k2) + log_iv(2 .* jj + r, k1) - k1 - k2);
            
        b = exp(log_iv(jj, k2) + log_iv(abs(2 .* jj - r), k1) - k1 - k2);
        
        term = a .* exp(+2.i .* jj .* delta) + ...
               b .* exp(-2.i .* jj .* delta);
           
        varphi_r = varphi_r + term;
        
        store_a = [store_a, a];
        store_b = [store_b, b];
        
    end
    disp(jj)
    T = 2 .* pi .* exp(1.i .* r .* m1 ) .* varphi_r;
    T = reshape(T, R, C);
end 