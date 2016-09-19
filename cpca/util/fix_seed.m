function fix_seed(seed)
% Function to fix the seed of the 
    try
        rng(seed);
    catch
        % Avoid errors in matlab versions older than R2014a
        %#ok<*RAND>
        rand('seed', seed); 
        randn('seed', seed);
    end
end