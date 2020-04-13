function [Wopt,lambdaopt,RMSEval,RMSEest] = multiframe_lasso_cv(T,X,lambdavec,K)
% [wopt,lambdaopt,VMSE,EMSE] = multiframe_lasso_cv(T,X,lambdavec,n)
% Calculates the LASSO solution for all frames and trains the
% hyperparameter using cross-validation.
%
%   Output:
%   Wopt        - mxnframes LASSO estimate for optimal lambda
%   lambdaopt   - optimal lambda value
%   VMSE        - vector of validation MSE values for lambdas in grid
%   EMSE        - vector of estimation MSE values for lambdas in grid
%
%   inputs:
%   T           - NNx1 data column vector
%   X           - NxM regression matrix
%   lambdavec   - vector grid of possible hyperparameters
%   K           - number of folds

% Define some sizes
NN = length(T);
[N,M] = size(X);
Nlam = length(lambdavec);

% Set indexing parameters for moving through the frames.
framehop = N;
idx = (1:N)';
framelocation = 0;
Nframes = 0;
while framelocation + N <= NN
    Nframes = Nframes + 1; 
    framelocation = framelocation + framehop;
    
end % Calculate number of frames.

% Preallocate
Wopt = zeros(M,Nframes);
SEval = zeros(K,Nlam);
SEest = zeros(K,Nlam);

% Set indexing parameter for the cross-validation indexing. FIX INDEXING
% WITH DIFFERENT FRAMES
Nval = floor(N/K);
cvhop = Nval;
randomind = randsample(size(X,1), size(X,1)); % Select random indices for picking out validation and estimation indices. 
    
framelocation = 0;
for kframe = 1:Nframes % First loop over frames
    
    cvlocation = 0;
%     if kframe == Nframes %If we are at the last frame, move back frame to include 352 data points
%         framelocation = size(T,1)-size(X,1);
%         %size_sample = size(T,1) - Nframes*size(X,1);
%         %randomind = randsample(size_sample,size_sample);
%         
%         
%     end
    
    for kfold = 1:K % Then loop over the folds
        
        valind = randomind(Nval*(kfold-1) + 1 : Nval*kfold); % Select validation indices
        estind = randomind; % Select estimation indices
        estind(Nval*(kfold-1) + 1 : Nval*kfold) = [];
        assert(isempty(intersect(valind,estind)), "There are overlapping indices in valind and estind!"); % assert empty intersection between valind and estind
    
        
        t = T(framelocation + idx); % Set data in this frame
        wold = zeros(M,1);  % Initialize old weights for warm-starting.
        
        for klam = 1:Nlam  % Finally loop over the lambda grid
            
            what = skeleton_lasso_ccd(t(estind),X(estind,:),lambdavec(klam), wold);% Calculate LASSO estimate at current frame, fold, and lambda
            
            SEval(kfold,klam) = SEval(kfold,klam) + 1/Nlam*norm(t(valind) - X(valind,:)*what); % Add validation error at current frame, fold and lambda to the validation error for this fold and lambda, summing tthe error over the frames
            SEest(kfold,klam) = SEest(kfold,klam) + 1/Nlam*norm(t(estind) - X(estind,:)*what); % Add estimation error at current frame, fold and lambda to the estimation error for this fold and lambda, summing tthe error over the frames
            
            wold = what; % Set current LASSO estimate as estimate for warm-starting.
            disp(['Frame: ' num2str(kframe) ', Fold: ' num2str(kfold) ', Hyperparam: ' num2str(klam)]) % Display progress through frames, folds and lambda-indices.
        end
        
        cvlocation = cvlocation+cvhop; % Hop to location for next fold.
    end
    
    framelocation = framelocation + framehop; % Hop to location for next frame.
    
end



MSEval = mean(SEval,1); % Average validation error across folds
MSEest = mean(SEest,1); % Average estimation error across folds

[~, index] = min(MSEval);
lambdaopt = lambdavec(index); % Select optimal lambda 

% Move through frames and calculate LASSO estimates using both estimation
% and validation data, store in Wopt.
framelocation = 0;
for kframe = 1:Nframes
    t = T(framelocation + idx);
    Wopt(:,kframe) = skeleton_lasso_ccd(t,X,lambdaopt,wold);
    framelocation = framelocation + framehop;
end

RMSEval = sqrt(MSEval);
RMSEest = sqrt(MSEest);

end

