% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);
% 数据集分为两类，x1 和x2
% 每类样本的特征值都是一维的
% disp(size(x1_train))
% disp(x1_train)
% disp('--------------------------------------------------')
% disp(size(x1_test))
% disp(x1_test)
% disp('--------------------------------------------------')
% disp(size(x2_train))
% disp(x2_train)
% disp('--------------------------------------------------')
% disp(size(x2_test))
% disp(x2_test)
% disp('--------------------------------------------------')
range = [min(all_x), max(all_x)];
% 一维特征值的范围
train_x = get_x_distribution(x1_train, x2_train, range);
% 统计出来的train训练集， 第一维是x1类的特征值分布情况，第二维是x2的
test_x = get_x_distribution(x1_test, x2_test, range);
% disp(size(train_x))
% disp(train_x)
% disp('--------------------------------------------------')
% disp(size(test_x))
% disp(test_x)
% disp('--------------------------------------------------')

%% Part1 likelihood: 
l = likelihood(train_x);

bar(range(1):range(2), l');
xlabel('x');
ylabel('P(x|\omega)');
axis([range(1) - 1, range(2) + 1, 0, 0.5]);

%TODO
%compute the number of all the misclassified x using maximum likelihood decision rule
[max_value, max_l_decision] = min(l);
[C, N_test] = size(test_x);
i = 0;
wrongnum= 0;
while i < N_test
    i = i + 1;
    wrongnum = wrongnum + test_x(max_l_decision(i), i);
end
% disp(max_l_decision)
% disp(max_arg)
% disp(test_x)
disp('maximum likelihood decision rule')
disp('wrongnum')
disp(wrongnum)
disp('test error rate')
[x1_1, x1_2] = size(x1_test);
[x2_1, x2_2] = size(x2_test);
num = x1_2 + x2_2;
disp(wrongnum/num);
%% Part2 posterior:
p = posterior(train_x);

figure;
bar(range(1):range(2), p');
xlabel('x');
ylabel('P(\omega|x)');
axis([range(1) - 1, range(2) + 1, 0, 1.2]);

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule
[max_value, max_l_decision] = min(p);
[C, N_test] = size(test_x);
i = 0;
wrongnum= 0;
while i < N_test
    i = i + 1;
    wrongnum = wrongnum + test_x(max_l_decision(i), i);
end
% disp(max_l_decision)
% disp(max_arg)
% disp(test_x)
disp('optimal bayes decision rule')
disp('wrongnum')
disp(wrongnum)
disp('test error rate')
disp(wrongnum/num)
%% Part3 risk:
risk = [0, 1; 2, 0];
%TODO
%get the minimal risk using optimal bayes decision rule and risk weights
lambda = [0, 1; 2, 0];
R_a_x = zeros(2, N_test);
for i = 1:N_test
    R_a_x(1, i) = lambda(1, 1) * p(1, i) + lambda(1, 2) * p(2, i);
    R_a_x(2, i) = lambda(2, 1) * p(1, i) + lambda(2, 2) * p(2, i);
end
disp('-------------------------------------------')
disp('get the minimal risk using optimal bayes decision rule and risk weights')
disp('conditional risk')
disp( R_a_x );

[max_value, max_l_decision] = max(R_a_x);
[C, N_test] = size(test_x);
i = 0;
wrongnum= 0;
while i < N_test
    i = i + 1;
    wrongnum = wrongnum + test_x(max_l_decision(i), i);
end
% disp(max_l_decision)
% disp(max_arg)
% disp(test_x)
disp('wrongnum')
disp(wrongnum)
disp('test error rate')
disp(wrongnum/num)
