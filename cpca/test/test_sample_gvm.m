function test_sample_gvm()
% test_sample_gvm
% test the sampler for the Generalised von Mises sampler by histogram
% inspection

    N = 4000;
    k1 = 10;
    k2 = 10;
    m1 = pi / 7 + 0;
    m2 = pi / 7 + pi / 2;
    
    kappa1 = k1 * ones(N, 1);
    kappa2 = k2 * ones(N, 1);
    nu1 = m1 * ones(N, 1);
    nu2 = m2 * ones(N, 1);
    
    s = sample_gvm(nu1, nu2, kappa1, kappa2);

    theta = linspace(0, 2 .* pi, 1E6);
    gvm = k1 .* cos(theta - m1) + k2 .* cos(2 .* (theta - m2));
    egvm = exp(gvm - max(gvm)) ./ sum(exp(gvm - max(gvm)));
    
    figure
    subplot(2,1,1)
    hist(s, N / 4)
    xlim([0, 2 .* pi])
    subplot(2,1,2)
    plot(theta, egvm)
    xlim([0, 2 .* pi])
end