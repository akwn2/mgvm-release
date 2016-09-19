function error = test_pack_as_array()
% test_pack_as_array
% Unit tests for the pack_as_array function

    % Create a basic problem setup
    N = 500;
    M = 4;
    D = 2;
    Y = rand(M, N);
    model = cpca(Y, D);

    %% Test 1: Simple variables
    % Create some variables to pack
    A = rand(M, D);
    kappa = rand(D, 1);
    prc2 = rand();
    
    model.p_A.xv = A;
    model.p_kappa.xv = kappa;
    model.p_prc2.xv = prc2;
    
    % Create a packing list and pack values
    model.name_map('Fq_variables') = ...
                                  {'p_A', ...
                                   'p_kappa', ...
                                   'p_prc2' ...
                                  };

    model.name_map('Fq_subset_ini') = ...
                                  {[1, 1], ...  %'p_A', ...
                                   1, ...       %'p_kappa', ...
                                   1, ...       %'p_prc2' ...
                                  };

    model.name_map('Fq_subset_end') = ...
                                  {[M, D], ...  %'p_A', ...
                                   D, ...       %'p_kappa', ...
                                   1 ...        %'p_prc2'...    
                                  };
    
    values = model.pack_as_array('variables','Fq');

    % Assert that it was correctly packed
    error = 0;
    try
        assert_equal(reshape(values(1:M * D), [M, D]), A,...
                    'Failed packing A');

        assert_equal(reshape(values(M * D + 1:M * D + D),[D, 1]), kappa,...
                    'Failed packing kappa');

        assert_equal(values(end), prc2, 'Failed packing sigma');
        
        fprintf('Test 1 passed.\n');
        
    catch Except
        fprintf('!!! ERROR FOUND IN TEST 1 !!!\n');
        
        fprintf(Except.identifier);
        fprintf(Except.message);
        error = error + 1;
    end
    
    
    %% Test 2: Simple derivatives
    % Create some derivatives to pack
    d_A = rand(M, D);
    d_kappa = rand(D, 1);
    d_prc2 = rand();
    
    model.p_A.dx = d_A;
    model.p_kappa.dx = d_kappa;
    model.p_prc2.dx = d_prc2;
    
    values = model.pack_as_array('gradients','Fq');
    
    % Assert that it was correctly packed
    try
        assert_equal(reshape(values(1:M * D), [M, D]), d_A,...
                     'Failed packing A');

        assert_equal(reshape(values(M * D + 1:M * D + D),[D, 1]),...
                     d_kappa,...
                     'Failed packing kappa');

        assert_equal(values(end), d_prc2, 'Failed packing sigma');
        
        fprintf('Test 2 passed.\n');
        
    catch Except
        fprintf('!!! ERROR FOUND IN TEST 2 !!!\n');
        
        fprintf(Except.identifier);
        fprintf(Except.message);
        error = error + 1;
    end
    
    
    %% Test 3: Simple lower bounds
    % Create some derivatives to pack
    lb_A = rand(M, D);
    lb_kappa = rand(D, 1);
    lb_prc2 = rand();
    
    model.p_A.lb = lb_A;
    model.p_kappa.lb = lb_kappa;
    model.p_prc2.lb = lb_prc2;
    
    values = model.pack_as_array('lower_bounds','Fq');
    
    % Assert that it was correctly packed
    try
        assert_equal(reshape(values(1:M * D), [M, D]), lb_A,...
                     'Failed packing A');

        assert_equal(reshape(values(M * D + 1:M * D + D),[D, 1]), ...
                     lb_kappa,...
                     'Failed packing kappa');

        assert_equal(values(end), lb_prc2, 'Failed packing sigma');
        
        fprintf('Test 3 passed.\n');
        
    catch Except
        fprintf('!!! ERROR FOUND IN TEST 3 !!!\n');
        
        fprintf(Except.identifier);
        fprintf(Except.message);
        error = error + 1;
    end
    
    
    %% Test 4: Simple upper bounds
    % Create some derivatives to pack
    ub_A = rand(M, D);
    ub_kappa = rand(D, 1);
    ub_prc2 = rand();
    
    model.p_A.ub = ub_A;
    model.p_kappa.ub = ub_kappa;
    model.p_prc2.ub = ub_prc2;
    
    values = model.pack_as_array('upper_bounds','Fq');
    
    % Assert that it was correctly packed
    try
        assert_equal(reshape(values(1:M * D), [M, D]), ub_A,...
                    'Failed packing A');

        assert_equal(reshape(values(M * D + 1:M * D + D),[D, 1]), ...
                     ub_kappa,...
                     'Failed packing kappa');

        assert_equal(values(end), ub_prc2, 'Failed packing sigma');
        
        fprintf('Test 4 passed.\n');
        
    catch Except
        fprintf('!!! ERROR FOUND IN TEST 4 !!!\n');
        
        fprintf(Except.identifier);
        fprintf(Except.message);
        error = error + 1;
    end
    
    
end