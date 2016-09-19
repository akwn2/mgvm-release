function error = test_entropy_gamma()
% test_entropy_gamma
% tests the entropy for the gamma distribution using grid integration
    
    tol = 1E-6;
    
    a = 10;
    b = 1;
    
    h_fcn = entropy_gamma(a, b);
    
    h_grid = entropy_gamma_grid(a, b);
    
    error = abs(h_fcn - h_grid) > tol;

    if error
        fprintf('-------------------------------------------\n');
        fprintf('!!!           TEST *NOT* PASSED         !!!\n');
        fprintf('-------------------------------------------\n');
    else
        fprintf('Test OK\n');
    end
end