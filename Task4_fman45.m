load A1_data.mat

lambda_grid = [0.1 10 5];

for i = 1:3
    what = skeleton_lasso_ccd(t,X,lambda_grid(i));
    idx = find(what ~= 0);
    size((what(idx)),1); %number of weights differing from zero
end

%%
%Plot t and estimate
l_1 = plot(n, t,'*');
hold on
l_2 = plot(n, X*what,'o');
l_3 = plot(ninterp, Xinterp*what);
legend('Data points', 'Estimation using new weights', 'Interpolated curve')






