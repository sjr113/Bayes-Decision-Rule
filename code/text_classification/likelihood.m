function l = likelihood(x)
%LIKELIHOOD Different Class Feature Liklihood 
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N matrix
%

[C, N] = size(x);
l = zeros(C, N);
%TODO
m = repmat(sum(x, 2), 1, N);
% 每一类的样本总个数是已知的，所以也可以直接除样本总个数400 800，不用再求和
l = x./m;
end
