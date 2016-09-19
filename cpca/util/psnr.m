function score = psnr(y_data, y_pred)
% psnr
% Prediction Signal-to-noise ratio: calculates the signal to noise ratio
% in decibels using the reference signal (data) y_data and the
% reconstruction signal (prediction) y_pred

    N = size(y_data, 1);
    
    signal = 0;
    noise  = 0;
    for nn = 1:N
        signal = signal + norm(y_data(nn, :));
        noise  = noise  + norm(y_data(nn, :) - y_pred(nn, :));
    end
    
    score = 10 * log(signal / noise);
end