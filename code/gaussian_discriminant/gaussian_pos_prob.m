function p = gaussian_pos_prob(X, Mu, Sigma, Phi)
%GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.
% 样本的总个数为N, 每个样本的特征维数是M
N = size(X, 2);
% 表示样本类别一共K类
K = length(Phi);
% p就是要求的posterior probability
p = zeros(N, K);

% Your code HERE
p_y = Phi;
p_x_y_k = zeros(1, K);
for n = 1:N
    p_x = 0;
    for i = 1:K
        value = 1/2/pi/sqrt(det(Sigma(:, :, i)));
        p_x_y_k(1,i) = value * exp(-1/2* (X(:, n)-Mu(:,i))'* inv(Sigma(:, :, i))* (X(:, n)-Mu(:,i)));   
        p_x = p_x + p_x_y_k(1,i) * p_y(1,i);
    end
    p(n, :) = p_x_y_k .* p_y/p_x;
end

