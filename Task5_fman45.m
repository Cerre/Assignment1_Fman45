load A1_data.mat
lambda_min = 0.01;
lambda_max = 15;
N_lambda = 20;
lambda_grid = exp( linspace( log(lambda_min), log(lambda_max), N_lambda));
K = 20;


[wopt,lambdaopt,RMSEval,RMSEest] = skeleton_lasso_cv(t,X,lambda_grid,K);
%% Plotting with the optimal values form the cross validation
l_1 = plot(n, t,'*');
hold on
l_2 = plot(n, X*wopt,'o');
l_3 = plot(ninterp, Xinterp*wopt);
legend('Original data', 'Reconstructed data', 'Interpolated curve')
idx = find(wopt ~= 0);
wopt(idx);
hold off

%% Plott the RMSEval, RMSEest and optimal lambda
figure;
rmse_val_plot2 = semilogx(lambda_grid, RMSEval);
hold on
rmse_val_plot = semilogx(lambda_grid, RMSEval, '*');

rmse_est_plot = semilogx(lambda_grid, RMSEest);
rmse_est_plot2 = semilogx(lambda_grid, RMSEest,'o');
x = linspace(min(min(RMSEval,RMSEest)), max(max(RMSEval, RMSEest)), 100);
lambda_opt_plot = semilogx(lambdaopt*ones(size(x)), x, '--');
legend('RMSE validation error', 'Lambda values','RMSE estimation error', 'Lambda values', 'Optimal lambda')


