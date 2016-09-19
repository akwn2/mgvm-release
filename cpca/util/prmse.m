function score = prmse(y_data, y_pred)
% prmse
% Prediction Root-Mean-Squared Error: calculates the error of the
% prediction signal y_pred and the noiseless signal y_data
    [N, M] = size(y_data);
    error = 0;
    for nn = 1:N
        error = error + norm(y_data(nn, :) - y_pred(nn, :)) .^ 2;
    end
    score = sqrt( error / (M * N) );
end