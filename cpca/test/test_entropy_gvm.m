function error = test_entropy_gvm()
% test_entropy_gvm
% tests the entropy for the Generalised von Mises distribution using grid
% integration
    
    tol = 1E-6;
    
    a = 10;
    b = 1;
    
    h_fcn = entropy_gvm(a, b);
    
    h_grid = entropy_gvm_grid(a, b);
    
    error = abs(h_fcn - h_grid) > tol;

    if error
        fprintf('-------------------------------------------\n');
        fprintf('!!!           TEST *NOT* PASSED         !!!\n');
        fprintf('-------------------------------------------\n');
    else
        fprintf('Test OK\n');
    end
end