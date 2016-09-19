function error = test_moments_gamma()
% test_moments_gamma
% tests the entropy for the gamma distribution using grid integration
    
    tol = 1E-6;
    
    a = 10;
    b = 1;
    
    m_fcn = moments_gamma(a, b);
    
    m_int = moments_gamma_grid(a, b);
    
    error = abs(m_fcn - m_int) > tol;

    if error
        fprintf('-------------------------------------------\n');
        fprintf('!!!           TEST *NOT* PASSED         !!!\n');
        fprintf('-------------------------------------------\n');
    else
        fprintf('Test OK\n');
    end
end