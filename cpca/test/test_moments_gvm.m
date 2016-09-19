function error = test_moments_gvm()
% test_moments_gvm
% tests the moments for the Generaliseed von Mises distribution using grid
% integration
    
    tol = 1E-6;
    
    a = 10;
    b = 1;
    
    m_fcn = moments_gvm(a, b);
    
    m_grid = moments_gvm_grid(a, b);
    
    error = abs(m_fcn - m_grid) > tol;

    if error
        fprintf('-------------------------------------------\n');
        fprintf('!!!           TEST *NOT* PASSED         !!!\n');
        fprintf('-------------------------------------------\n');
    else
        fprintf('Test OK\n');
    end
end