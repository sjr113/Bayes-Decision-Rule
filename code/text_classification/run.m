%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
% 大小为1*N,N为words的个数
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
% 大小为1*N,N为words的个数
%N is the size of vocabulary.N是字典的大小
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;  % 非垃圾邮件，为负样本 y = 0
num_spam_train = 3372; % 垃圾邮件的个数，注意垃圾邮件为正样本  y = 1
%Do smoothing  拉普拉斯平滑
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);  % 把常规矩阵转换为稀疏矩阵
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;
% disp('-----------------------------------')
% disp(spam_test(3,:))
% disp('-----------------------------------')
%TODO
%Implement a ham/spam email classifier, and calculate the accuracy of your classifier

% 首先需要求正负样本中各个单词的概率, 注意要使用拉普拉斯平滑
% laplace平滑就是假设所有的单词之前都出现过一次
% ？？？？？不明白下面的公式到底是加2还是加N
phi = zeros(2, N);
log_phi = zeros(2, N);
% phi(1, :) = (ham_train + 1) / (num_ham_train + 2);  % 每个单词在ham中占的比例
phi(1, :) = (ham_train + 1) / (sum(ham_train) + N);  % 每个单词在ham中占的比例
log_phi(1, :) = log(phi(1, :)) ./ (ham_train + 1);
% 即在y = 0的样本中， 特征xj=1的比例
% phi(2, :) = (spam_train + 1) / (num_spam_train + 2); % 每个单词在spam中占的比例
phi(2, :) = (spam_train + 1) / (sum(spam_train) + N); % 每个单词在spam中占的比例
log_phi(2, :) = log(phi(2, :)) ./ (spam_train + 1);  % 因为已经假设过每个单词都出现过一次了，所以这里要加1
% 即在y = 1的样本中， 特征xj=1的比例

[words, index] = textread('all_word_map.txt', '%s%d');

disp('前十个能表征spam的单词')
[sort_words, top_10_word_index] = sort((phi(2, :)./phi(1, :)), 'descend');
top_index =  top_10_word_index(1:10);
% disp(top_index);
disp(words(top_index));

% 求正负样本占的比例
phi_y = zeros(2,1);
% phi_y(1, 1) =  num_ham_train / (num_ham_train + num_spam_train); % 负样本占的比例
phi_y(1, 1) =  sum(ham_train + 1) / (sum(ham_train +1) + sum(spam_train+1)); % 负样本占的比例
% phi_y(1, 1) =  sum(ham_train ) / (sum(ham_train) + sum(spam_train))
phi_y(2, 1) = 1 - phi_y(1, 1); % 正样本占的比例

% 由于假设条件独立，即 p(x1,x2...x5000 | y) = p(x1|y)*p(x2|y)*...*p(x5000|y)
% log_p_y_x = zeros(2, 1);
% % 这里采用取log求和的方式，来避免乘法，提高计算效率
% log_p_y_x(1, 1) = sum(log(phi(1, :))) + phi_y(1, 1);
% log_p_y_x(2, 1) = sum(log(phi(2, :))) + phi_y(2, 1);

% 当给定一个新的单词，计算它的条件概率p_spam_w 和p_ham_W
p_s_W = zeros(2, N);
p_s_W(1, :) = log_phi(1, :) + log(phi_y(1, 1));
p_s_W(2, :) = log_phi(2, :) + log(phi_y(2, 1));
% disp('111111111111111111111111111111111')
% disp(spam_train(1, 1))
% disp(spam_train(1, 2))
% disp(phi(2, 1))
% disp(phi(2, 2))
% disp(log(phi(2, 1)))
% disp(log(phi(2, 2)))
% disp(log_phi(2, 1))
% disp(log_phi(2, 2))
% disp(log(phi_y(1, 1)))
% disp(log(phi_y(2, 1)))
% disp('111111111111111111111111111111111')
% p_s_W(1, :) = p_s_W(1, :) ./ sum(p_s_W, 1);
% p_s_W(2, :) = p_s_W(2, :) ./ sum(p_s_W, 1);

% 下面读入test集合，进行判断
TP = 0;
FP = 0;
FN = 0;
TN = 0;
[P, N] = size(ham_test);
[Q, N ] = size(spam_test);

% for i = 1:Q
%     % 返回spam_test(i, :)中所有非0元素的索引 
%     ind = find(spam_test(i, :));
% %     ham_sum = sum(log(p_s_W(1, ind)).*spam_test(i,ind));
% %     spam_sum = sum(log(p_s_W(2, ind)).*spam_test(i,ind));
%     ham_sum = sum(p_s_W(1, ind).*spam_test(i,ind));
%     spam_sum = sum(p_s_W(2, ind).*spam_test(i,ind));
%     if ham_sum > spam_sum
%         FN = FN +1;
%     else
%         TP = TP +1;
%     end
% end

for i = 1:Q
    ham_sum = sum(p_s_W(1, :).*(spam_test(i,:) ));
    spam_sum = sum(p_s_W(2, :).*(spam_test(i,:)));
    if ham_sum > spam_sum
        FN = FN +1;
    else
        TP = TP +1;
    end
end

% for i = 1:P
%     % 返回ham_test(i, :)中所有非0元素的索引 
%     ind = find(ham_test(i, :));
% %     disp(ind)
% %     disp(ham_test(i,ind))
% %     disp(log(p_s_W(1, ind)))
% %     disp(log(p_s_W(2, ind)))
% %     ham_sum = sum(log(p_s_W(1, ind)).*ham_test(i,ind));
% %     spam_sum = sum(log(p_s_W(2, ind)).*ham_test(i,ind));
%     ham_sum = sum((p_s_W(1, ind)).*ham_test(i,ind));
%     spam_sum = sum((p_s_W(2, ind)).*ham_test(i,ind));
% %     disp(ham_sum)
% %     disp(spam_sum)
% %     disp(p_s_W(2, ind))
%     if ham_sum > spam_sum
%         TN = TN +1;
%     else
%         FP = FP +1;
%     end
% end

for i = 1:P
    ham_sum = sum(p_s_W(1, :).*(ham_test(i,:) ));
    spam_sum = sum(p_s_W(2, :).*(ham_test(i,:) ));
    if ham_sum > spam_sum
        TN = TN +1;
    else
        FP = FP +1;
    end
end

% accu = (TN + TP) / (P+Q)
accu = (TN + TP) / (TN + TP + FP + FN); 
disp(TN);
disp(TP);
disp( FP);
disp( FN);
disp('accuracy: ')
disp(accu);

precision = TP /(TP + FP);
recall = TP /(TP + FN);
disp('precision: ')
disp(precision)
disp('recall: ')
disp(recall)
    





