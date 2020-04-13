çload A1_data.mat

%Divide training data into 40 ms parts
%Use Xaudio for LASSO

lambda_min = 0.0001;
lambda_max = 5;
N_lambda = 10;
lambda_grid = exp( linspace( log(lambda_min), log(lambda_max), N_lambda));
K = 8;


[wopt,lambdaopt,RMSEval,RMSEest] = skeleton_multiframe_lasso_cv(Ttest,Xaudio,lambda_grid,K);


%% Get a denoised signal
[Yclean] = lasso_denoise(Ttest,Xaudio,lambdaopt);

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



