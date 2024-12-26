clc; clear;

%% Import data
data3D = [
    6, 3, 4, 4, 4, 5, 3, 2, 6, 5, 6, 5, 4, 5, 7, 8;
    4, 4, 5, 3, 3, 5, 4, 1, 2, 2, 4, 2, 6, 2, 5, 5;
    4, 6, 3, 4, 2, 4, 2, 3, 5, 3, 4, 5, 5, 6, 4, 8;
    5, 2, 3, 4, 2, 4, 3, 2, 5, 4, 3, 3, 6, 5, 5, 7
];

%% Parameters
lineWidth = 1;
fontSize = 12;
fileName = 'comparisonSubplots';
pictureSize = [200, 200, 1000, 800];
pictureResolution = '-r600';

%% Prepare colors (using parula colormap)
colormapLines = parula(size(data3D, 2)); % Use parula colormap

%% Create subplots
fig = figure;
set(gcf, 'Position', pictureSize);

% Dimension labels
dimLabels = {'Clarity', 'Contrast', 'Naturalness', 'Consistency'};

for dim = 1:4
    subplot(2, 2, dim); % 2x2 grid of subplots
    
    % Plot bar chart for current dimension
    bars = bar(data3D(dim, :), 'FaceColor', 'flat'); % Use 'flat' to allow per-bar color customization
    
    % Apply consistent colors
    bars.CData = colormapLines; % Assign different color to each bar
    
    % Customize subplot
    title(dimLabels{dim}, 'FontSize', fontSize, 'FontName', 'Times New Roman');
    ylabel('Average Scores', 'FontSize', fontSize, 'FontName', 'Times New Roman');
    xticks(1:16);
    xticklabels({'IFCNN', 'PMGI', 'CU-Net', 'U2Fusion', 'MFF-GAN', 'SDNet', ...
                 'SwinFusion', 'DeFusion', 'ZMFF', 'MGDN', 'MUFusion', ...
                 'PSLPT', 'DB-MFIF', 'DeepM$^2$CDL', 'TC-MoA', 'NSNPFuse'}); 
    xtickangle(-90);
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', fontSize, 'FontName', 'Times New Roman'); % Set Times New Roman for xticklabels
    ylim([0, 10]); % Set Y-axis limits consistent with 3D plot
    
    % Display the values on top of each bar (only integers)
    for i = 1:length(data3D(dim, :))
        % Place the text slightly above each bar
        text(i, data3D(dim, i) + 0.1, num2str(data3D(dim, i), '%d'), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', fontSize, 'FontName', 'Times New Roman');
    end
end

% Remove white margins
set(gcf, 'Units', 'inches'); % Set figure size in inches for precise control
pos = get(gcf, 'Position'); % Get current figure size
set(gcf, 'PaperPositionMode', 'auto', 'PaperUnits', 'inches', 'PaperSize', [pos(3), pos(4)]);

% Save the figure
print(fileName, '-djpeg', pictureResolution);
