function error = test_unpack_from_array()
% test_unpack_from_array
% Unit tests for the unpack_from_array function

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
    % Packing
    values = model.pack_as_array('variables','Fq');
    
    % Unpacking
    
    model2 = model.unpack_from_array(values, 'variables', 'Fq');

    % Assert that it was correctly packed
    error = 0;
    try
        assert_equal(model2.p_A.xv, A, ...
                    'Failed unpacking A\n');

        assert_equal(model2.p_kappa.xv, kappa, ...
                    'Failed unpacking kappa\n');

        assert_equal(model2.p_prc2.xv, prc2, ...
                    'Failed unpacking sigma\n');
        
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
    
    % Unpacking
    
    model2 = model.unpack_from_array(values, 'gradients', 'Fq');
    
    % Assert that it was correctly packed
    try
        assert_equal(model2.p_A.dx, d_A, ...
                    'Failed unpacking A\n');

        assert_equal(model2.p_kappa.dx, d_kappa,...
                    'Failed unpacking kappa\n');

        assert_equal(model2.p_prc2.dx, d_prc2, ...
                    'Failed unpacking sigma\n');
        
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
    
    % Unpacking
    
    model2 = model.unpack_from_array(values, 'gradients', 'Fq');
    
    % Assert that it was correctly packed
    try
        assert_equal(model2.p_A.lb, lb_A, ...
                    'Failed unpacking A\n');

        assert_equal(model2.p_kappa.lb, lb_kappa, ...
                    'Failed unpacking kappa\n');

        assert_equal(model2.p_prc2.lb, lb_prc2, ...
                    'Failed unpacking sigma\n');
        
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
    
    % Unpacking
    
    model2 = model.unpack_from_array(values, 'gradients', 'Fq');
    
    % Assert that it was correctly packed
    try
        assert_equal(model2.p_A.ub, ub_A, ...
                    'Failed unpacking A\n');

        assert_equal(model2.p_kappa.ub, ub_kappa, ...
                    'Failed unpacking kappa\n');

        assert_equal(model2.p_prc2.ub, ub_prc2, ...
                    'Failed unpacking sigma\n');
        
        fprintf('Test 4 passed.\n');
        
    catch Except
        fprintf('!!! ERROR FOUND IN TEST 4 !!!\n');
        
        fprintf(Except.identifier);
        fprintf(Except.message);
        error = error + 1;
    end
    
    
end