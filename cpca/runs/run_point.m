function errcode = run_point(dataset, D, noise, seed, solver, greedy)
% run_point.m
% Function to standardise the procedure for running tests with different
% models
    if nargin < 6
        greedy = true;
    end

    fix_seed(seed);
    
    case_name = [dataset, '_point_', num2str(D), '_',...
                 num2str(noise),'_',num2str(seed)];
    
    fprintf('---------------------------------\n');
    fprintf('      STARTED RUNNING CASE       \n');
    fprintf('---------------------------------\n');

    fprintf('Loading data... ');
    try
        data = load(['./data/', dataset, '.mat']);
    catch
        % Display end message
        fprintf('!!! ERROR - could not open dataset.\n');
        errcode = -1;
        return
    end
    fprintf('Ok!\n');
    
    % Create data splits
    Y = zeros(data.N, data.M);
    Y(:, 1:2:end) = data.mocap_x;
    Y(:, 2:2:end) = data.mocap_y;

    [N_train, Y_train, N_test, Y_test] = split_data(Y, 0.5, true);

    options = {...
        {'name', case_name}, ...
        {'opt_max_iter', 5E3}, ...
        {'opt_tol', 1E-6}, ...
        {'opt_solver', solver}, ...
        };
        
    Y_train = (Y_train + noise .* randn(size(Y_train)))';
    Y_held = (Y_test + noise .* randn(size(Y_test)))';
    
    if greedy
        model_train = greedy_init(Y_train, D, options);
    else
        model_train = Cpca(Y_train, D, options);
        model_train = model_train.preset_opt_free_point_vm();
    end

    model_train = model_train.optimize_free_energy();

    % Testing by denoising
    model_test = Cpca(Y_held, D, options);
    model_test = model_test.preset_opt_free_point_vm();

    model_test.p_A.xv = model_train.p_A.xv;
    model_test.p_B.xv = model_train.p_B.xv;
    model_test.p_kappa.xv = model_train.p_kappa.xv;
    model_test.p_prc2.xv = model_train.p_prc2.xv;

    model_test = model_test.preset_opt_free_point_vm_mf();
    model_test = model_test.optimize_free_energy();
    model_test = model_test.get_moments_q_phi(false);

    Y_pred = model_test.p_A.xv * model_test.m_cos_phi.xv + ...
             model_test.p_B.xv * model_test.m_sin_phi.xv;

    point_rmse = prmse(Y_test, Y_pred');
    point_snr = psnr(Y_test, Y_pred');
    
    fprintf('Point CPCA:\n')
    fprintf(' RMSE: %1.4e\n', point_rmse);
    fprintf(' SNR:  %1.4e\n', point_snr);
    
    save([case_name, '.mat'],'model_train','model_test',...
         'point_rmse','point_snr', 'Y_train', 'Y_test', 'Y_held');

    fprintf('---------------------------------\n');
    fprintf('     FINISHED RUNNING CASE       \n');
    fprintf('---------------------------------\n');
end