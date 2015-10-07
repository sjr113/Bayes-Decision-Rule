%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
% ��СΪ1*N,NΪwords�ĸ���
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
% ��СΪ1*N,NΪwords�ĸ���
%N is the size of vocabulary.N���ֵ�Ĵ�С
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;  % �������ʼ���Ϊ������ y = 0
num_spam_train = 3372; % �����ʼ��ĸ�����ע�������ʼ�Ϊ������  y = 1
%Do smoothing  ������˹ƽ��
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);  % �ѳ������ת��Ϊϡ�����
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;
% disp('-----------------------------------')
% disp(spam_test(3,:))
% disp('-----------------------------------')
%TODO
%Implement a ham/spam email classifier, and calculate the accuracy of your classifier

% ������Ҫ�����������и������ʵĸ���, ע��Ҫʹ��������˹ƽ��
% laplaceƽ�����Ǽ������еĵ���֮ǰ�����ֹ�һ��
% ��������������������Ĺ�ʽ�����Ǽ�2���Ǽ�N
phi = zeros(2, N);
log_phi = zeros(2, N);
% phi(1, :) = (ham_train + 1) / (num_ham_train + 2);  % ÿ��������ham��ռ�ı���
phi(1, :) = (ham_train + 1) / (sum(ham_train) + N);  % ÿ��������ham��ռ�ı���
log_phi(1, :) = log(phi(1, :)) ./ (ham_train + 1);
% ����y = 0�������У� ����xj=1�ı���
% phi(2, :) = (spam_train + 1) / (num_spam_train + 2); % ÿ��������spam��ռ�ı���
phi(2, :) = (spam_train + 1) / (sum(spam_train) + N); % ÿ��������spam��ռ�ı���
log_phi(2, :) = log(phi(2, :)) ./ (spam_train + 1);  % ��Ϊ�Ѿ������ÿ�����ʶ����ֹ�һ���ˣ���������Ҫ��1
% ����y = 1�������У� ����xj=1�ı���

[words, index] = textread('all_word_map.txt', '%s%d');

disp('ǰʮ���ܱ���spam�ĵ���')
[sort_words, top_10_word_index] = sort((phi(2, :)./phi(1, :)), 'descend');
top_index =  top_10_word_index(1:10);
% disp(top_index);
disp(words(top_index));

% ����������ռ�ı���
phi_y = zeros(2,1);
% phi_y(1, 1) =  num_ham_train / (num_ham_train + num_spam_train); % ������ռ�ı���
phi_y(1, 1) =  sum(ham_train + 1) / (sum(ham_train +1) + sum(spam_train+1)); % ������ռ�ı���
% phi_y(1, 1) =  sum(ham_train ) / (sum(ham_train) + sum(spam_train))
phi_y(2, 1) = 1 - phi_y(1, 1); % ������ռ�ı���

% ���ڼ��������������� p(x1,x2...x5000 | y) = p(x1|y)*p(x2|y)*...*p(x5000|y)
% log_p_y_x = zeros(2, 1);
% % �������ȡlog��͵ķ�ʽ��������˷�����߼���Ч��
% log_p_y_x(1, 1) = sum(log(phi(1, :))) + phi_y(1, 1);
% log_p_y_x(2, 1) = sum(log(phi(2, :))) + phi_y(2, 1);

% ������һ���µĵ��ʣ�����������������p_spam_w ��p_ham_W
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

% �������test���ϣ������ж�
TP = 0;
FP = 0;
FN = 0;
TN = 0;
[P, N] = size(ham_test);
[Q, N ] = size(spam_test);

% for i = 1:Q
%     % ����spam_test(i, :)�����з�0Ԫ�ص����� 
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
%     % ����ham_test(i, :)�����з�0Ԫ�ص����� 
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
    





