clc
clear

%创建一个3x3的空figure
figure('Units','centimeters','Position',[8 5 30 20]);

ylim_max=0.2;
subplot(3,3,1)
hold on
h1_train=plot(NaN,NaN);
h1_valid=plot(NaN,NaN);
title('q=10%');
legend('train','valid','Location','northeast');
xlim([0,200]);
xlabel('epoch');
ylabel('loss');
grid on
ylim([0,ylim_max])

subplot(3,3,2)
hold on
h2_train=plot(NaN,NaN);
h2_valid=plot(NaN,NaN);
title('q=20%');
legend('train','valid','Location','northeast');
xlim([0,200]);
xlabel('epoch');
ylabel('loss');
grid on
ylim([0,ylim_max])

subplot(3,3,3)
hold on
h3_train=plot(NaN,NaN);
h3_valid=plot(NaN,NaN);
title('q=30%');
legend('train','valid','Location','northeast');
xlim([0,200]);
xlabel('epoch');
ylabel('loss');
grid on
ylim([0,ylim_max])

subplot(3,3,4)
hold on
h4_train=plot(NaN,NaN);
h4_valid=plot(NaN,NaN);
title('q=40%');
legend('train','valid','Location','northeast');
xlim([0,200]);
xlabel('epoch');
ylabel('loss');
grid on
ylim([0,ylim_max])

subplot(3,3,5)
hold on
h5_train=plot(NaN,NaN);
h5_valid=plot(NaN,NaN);
title('q=50%');
legend('train','valid','Location','northeast');
xlim([0,200]);
xlabel('epoch');
ylabel('loss');
grid on
ylim([0,ylim_max])

subplot(3,3,6)
hold on
h6_train=plot(NaN,NaN);
h6_valid=plot(NaN,NaN);
title('q=60%');
legend('train','valid','Location','northeast');
xlim([0,200]);
xlabel('epoch');
ylabel('loss');
grid on
ylim([0,ylim_max])

subplot(3,3,7)
hold on
h7_train=plot(NaN,NaN);
h7_valid=plot(NaN,NaN);
title('q=70%');
legend('train','valid','Location','northeast');
xlim([0,200]);
xlabel('epoch');
ylabel('loss');
grid on
ylim([0,ylim_max])

subplot(3,3,8)
hold on
h8_train=plot(NaN,NaN);
h8_valid=plot(NaN,NaN);
title('q=80%');
legend('train','valid','Location','northeast');
xlim([0,200]);
xlabel('epoch');
ylabel('loss');
grid on
ylim([0,ylim_max])

subplot(3,3,9)
hold on
h9_train=plot(NaN,NaN);
h9_valid=plot(NaN,NaN);
title('q=90%');
legend('train','valid','Location','northeast');
xlim([0,200]);
xlabel('epoch');
ylabel('loss');
grid on
ylim([0,ylim_max])

Q=10:10:100;
H_train=[h1_train,h2_train,h3_train,h4_train,h5_train,h6_train,h7_train,h8_train,h9_train];
H_valid=[h1_valid,h2_valid,h3_valid,h4_valid,h5_valid,h6_valid,h7_valid,h8_valid,h9_valid];

while true
    parfor i=1:9 
        %读取training_log/wind_q{q}.txt文件 并实时绘制曲线图
        path_train_loss = strcat('training_log/wind_q',num2str(Q(i)),'/train_loss.txt');
        path_valid_loss = strcat('training_log/wind_q',num2str(Q(i)),'/val_loss.txt');
        train_loss = readmatrix(path_train_loss);  % 读取文件中的数据
        val_loss = readmatrix(path_valid_loss);  % 读取文件中的数据
        
        %更新数据
        set(H_train(i), 'XData', 1:length(train_loss), 'YData', train_loss);
        set(H_valid(i), 'XData', 1:length(val_loss), 'YData', val_loss);
   
        %更新figure中的每个axes
        drawnow;
    end
end