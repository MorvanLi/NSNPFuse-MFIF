% clc;clear
% %% Import data
% data3D=xlsread('data3D.xlsx');
% %% Parameters
% lineWidth=1;
% fontSize=28;
% fileName='histogram3D';
% pictureSize=[200,200,900,750];
% pictureResolution='-r300';
% %% Plot
% hist3(data3D,'Nbins',[9 9],'CDataMode','auto','FaceColor','interp')
% box on
% %% Seeting
% set(gca,'LineWidth',lineWidth)
% xlabel('Force [N]');
% ylabel('Velocity [m/s]');
% zlabel('Frequency');
% set(gca,'FontSize',fontSize,'Fontname', 'Times New Roman');
% set(gca,'XLim',[-3,3],'YLim',[-3,3],'ZLim',[0,8]);
% set(gca,'xtick',[-3:1:3],'ytick',[-3:1:3],'ztick',[0:2:8]);
% set(gcf,'Position',pictureSize)
% print(fileName,'-dtiff',pictureResolution);


clc; clear;

%% Import data 
data3D = [
    6, 3, 4, 4, 4, 5, 5, 2, 6, 5, 6, 5, 4, 5, 7, 8;
    4, 4, 6, 6, 3, 5, 4, 4, 2, 2, 4, 2, 6, 2, 5, 5;
    4, 6, 4, 4, 5, 4, 2, 3, 5, 3, 4, 5, 5, 6, 4, 8;
    5, 2, 3, 4, 2, 4, 3, 2, 5, 4, 3, 3, 6, 6, 6, 7
];

%% Parameters
lineWidth = 1;
fontSize = 16;
fileName = 'comparison3D';
pictureSize = [200, 200, 1000, 500];
pictureResolution = '-r600';

%% Plot
% 绘制3D直方图
bar3(data3D);  % 使用 bar3 来绘制3D柱状图

%% Setting
set(gca, 'LineWidth', lineWidth)
% xlabel('Comparison Methods');
% ylabel('Dimensions');
zlabel('Average Scores');
set(gca, 'FontSize', fontSize, 'Fontname', 'Times New Roman');
set(gca, 'XLim', [0.5 16.5], 'YLim', [0.5 4.5], 'ZLim', [1, 10]);  % 假设得分范围为0到1
set(gca, 'xtick', 1:16, 'ytick', 1:4);  % 设置X轴和Y轴的刻度

% Set Y-axis labels to the 4 dimensions
yticklabels({'Clarity', 'Contrast', 'Naturalness', 'Consistency'});
ytickangle(25);  % 将X轴标签旋转45度
% 设置X轴标签为指定的对比方法
xticklabels({'IFCNN', 'PMGI', 'CU-Net', 'U2Fusion', 'MFF-GAN', 'SDNet', ...
             'SwinFusion', 'DeFusion', 'ZMFF', 'MGDN', 'MUFusion', ...
             'PSLPT', 'DB-MFIF', 'DeepM$^2$CDL', 'TC-MoA', 'NSNPFuse (Ours)'});
set(gca, 'TickLabelInterpreter', 'latex');  % 设置刻度标签解释器为 LaTeX       

xtickangle(-35);  % 将X轴标签旋转45度
% 设置图形大小
set(gcf, 'Position', pictureSize)

% Adjust the 3D view (azimuth and elevation)
% view(5, 10);  % View from 45 degrees azimuth and 30 degrees elevation
view(30, 20);  % 调整为更合适的俯视角度和方位角
% 保存图片
% print(fileName, '-dtiff', pictureResolution);
print(fileName, '-djpeg', pictureResolution);












