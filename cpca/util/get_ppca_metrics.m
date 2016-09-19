function [ppca_rmse, ppca_snr] = get_ppca_metrics(D, Y_train, Y_held, Y_true)
% get_ppca_metrics
%   calculates the rmse and snr for the ppca case

    % Train PPCA
    %------------------------------------------------
    opt = statset('ppca'); opt.MaxIter = 5000;
    
    % Train the model
    [~,~,~,mu,~,S] = ppca(Y_train, D,'Options', opt);

    % Predict using PPCA
    %------------------------------------------------
    
    % Manually extract the means to make test data zero-mean
    % (so that PPCA will not learn the means)
    N_held = size(Y_held, 1);
    mu_held = mean(Y_train, 1);
    Y_held = Y_held - repmat(mu_held, [N_held, 1]);
    
    % Do a sigle E-step with the learned parameters
    opt = statset('ppca'); opt.MaxIter = 1;
    
    [~,~,~,~,~,Spred] = ppca(Y_held, D,'Options', opt, ...
                             'W0', S.W, 'v0', S.v);
                         
    predictions = Spred.Xexp * S.W' + repmat(mu, [N_held, 1]);
    
    % Calculate RMSE and SNR
    ppca_rmse = prmse(Y_true, predictions);
    ppca_snr = psnr(Y_true, predictions);
end

