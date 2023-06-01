clc; clear;

% Load the embeddings, labels, and video
tD_embeddings = readmatrix("D:\Downloads\embeds_2d.csv");
embeddings = readmatrix("D:\Downloads\embeds_3d.csv");  % Assuming embeddings.mat contains the 1000x3 matrix of embeddings
labels = readmatrix("D:\Downloads\labels.csv");  % Assuming labels.mat contains the 1000x1 vector of labels
video = VideoReader('F:\DLC\final\Data\Test103-1DLC_resnet50_Test6.0Dec14shuffle1_200000_labeled.mp4');
% Assuming video.mp4 is the input video file
outDir = 'C:\Users\Eminent\Desktop\output.gif';

%%
% If you want MP4 as your output format:
clc; close all;

% Set up the figure and axes for plotting
fig = figure('Color', 'k');
set(fig, 'Position', [100, 100, 1500, 978]);  % Adjust the figure size as desired
ax1 = subplot(2, 2, [1, 3]);  % Left subplot for displaying the video frames
ax2 = subplot(2, 2, 2);  % Middle subplot for displaying the 3D embeddings
ax3 = subplot(2, 2, 4);  % Right subplot for displaying the 2D embeddings

% Create a custom colormap resembling 'magma' using 'parula'
parulaColormap = parula(256);
magmaColormap = parulaColormap(30:255, :);

% Plot the video frames
frameWidth = video.Width / 2; % Divide the frame width in half to accommodate the 3D and 2D plots
frameHeight = video.Height;
imshow(zeros(frameHeight, frameWidth, 3), 'Parent', ax1); % Display a black background
set(ax1, 'Color', [0.1 0.1 0.1]); % Set the background color of ax1 to dark gray
set(ax1, 'XColor', '#66ff99', 'YColor', '#66ff99'); % Set the axis numbers color to light green

% Plot the 3D embeddings with a fade
scatter3(ax2, embeddings(:, 1), embeddings(:, 2), embeddings(:, 3), 30, labels, 'filled', 'MarkerFaceAlpha', 0.1);
colormap(ax2, magmaColormap);
colorbar(ax2, 'Ticks', unique(labels), 'TickLabels', cellstr(num2str(unique(labels))), 'Color', 'w');
title(ax2, '3D Embeddings');
set(ax2, 'Color', [0.1 0.1 0.1]);  % Set the background color of ax2 to dark gray
set(ax2, 'XColor', '#66ff99', 'YColor', '#66ff99', 'ZColor', '#66ff99');  % Set the axis numbers color to light green
view(ax2, 3);  % Set the 3D view

% Plot the 2D embeddings with a fade
scatter(ax3, tD_embeddings(:, 1), tD_embeddings(:, 2), 30, labels, 'filled', 'MarkerFaceAlpha', 0.1);
colormap(ax3, magmaColormap);
colorbar(ax3, 'Ticks', unique(labels), 'TickLabels', cellstr(num2str(unique(labels))), 'Color', 'w');
title(ax3, '2D Embeddings');
set(ax3, 'Color', [0.1 0.1 0.1]);  % Set the background color of ax3 to dark gray
set(ax3, 'XColor', '#66ff99', 'YColor', '#66ff99');  % Set the axis numbers color to light green

% Add a title text box
t = 'PoseModel: Automated Behavioral State Estimation - model: GAT-50, IPM-SCS';
annotation(fig, 'textbox', [0.1 0.9 0.8 0.1], 'String', t, ...
    'Color', 'w', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 19, 'FontWeight', 'bold');

% Set up video parameters
fps = 30;
delayTime = 1 / fps;
outputVideo = VideoWriter(outDir, 'MPEG-4');
outputVideo.FrameRate = fps;
open(outputVideo);

% Create text annotations for time, frame number, and state
textAnnotation1 = annotation(fig, 'textbox', [0.035 0.75 0.11 0.05], 'String',...
    '', 'Color', 'w', 'EdgeColor', 'w','FontSize',13, 'fontweight', 'bold');
textAnnotation2 = annotation(fig, 'textbox', [0.035 0.70 0.11 0.05], 'String',...
    '', 'Color', 'w', 'EdgeColor', 'w', 'FontSize',13, 'fontweight', 'bold');
textAnnotation3 = annotation(fig, 'textbox', [0.035 0.65 0.11 0.05], 'String',...
    '', 'Color', 'w', 'EdgeColor', 'w','FontSize',13, 'fontweight', 'bold');

% Capture frames and write to the output video
for i = 1:25000%length(labels)
    % Read the video frame
    frame = readFrame(video);

    % Display the video frame
    imshow(frame, 'Parent', ax1);
    
    % Remove the previous current state dot in the 3D plot
    if i > 1
        delete(findobj(ax2, 'MarkerEdgeColor', 'r'));  % Delete the previous red dots
    end
    
    % Remove the previous current state dot in the 2D plot
    if i > 1
        delete(findobj(ax3, 'MarkerEdgeColor', 'r'));  % Delete the previous red dots
    end
    
    % Update the current embedding dot in the 3D plot
    hold(ax2, 'on');
    scatter3(ax2, embeddings(i, 1), embeddings(i, 2), embeddings(i, 3), 60, 'w', 'filled', 'MarkerEdgeColor', 'r');
    scatter3(ax2, embeddings(i, 1), embeddings(i, 2), embeddings(i, 3), 90, 'w', 'filled', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);
    hold(ax2, 'off');
    
    % Update the current embedding dot in the 2D plot
    hold(ax3, 'on');
    scatter(ax3, tD_embeddings(i, 1), tD_embeddings(i, 2), 60, 'w', 'filled', 'MarkerEdgeColor', 'r');
    scatter(ax3, tD_embeddings(i, 1), tD_embeddings(i, 2), 90, 'w', 'filled', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);
    hold(ax3, 'off');
    
    % Update the text annotations
    set(textAnnotation1, 'String', sprintf('Time: %.2f s', i/30));
    set(textAnnotation2, 'String', sprintf('Frame: %d', i));
    set(textAnnotation3, 'String', sprintf('State: %s', num2str(labels(i))));
    
    % Capture the current figure as a frame for the GIF
    frame = getframe(fig);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    % Write the frame to the GIF file
    if i == 1
        imwrite(imind, cm, outDir, 'gif', 'Loopcount', inf, 'DelayTime', delayTime);
    else
        imwrite(imind, cm, outDir, 'gif', 'WriteMode', 'append', 'DelayTime', delayTime);
    end
    % Write the current frame to the output video
    frame = getframe(fig);
    writeVideo(outputVideo, frame);
end

% Close the output video
close(outputVideo);

disp('MP4 file created successfully.');

%%
% If you need GIF as your output format:
clc; close all;

% Set up the figure and axes for plotting
fig = figure('Color', 'k');
set(fig, 'Position', [100, 100, 1200, 780]);  % Adjust the figure size as desired
ax1 = subplot(2, 2, [1, 3]);  % Left subplot for displaying the video frames
ax2 = subplot(2, 2, 2);  % Middle subplot for displaying the 3D embeddings
ax3 = subplot(2, 2, 4);  % Right subplot for displaying the 2D embeddings

% Create a custom colormap resembling 'magma' using 'parula'
parulaColormap = parula(256);
magmaColormap = parulaColormap(30:255, :);

% Plot the video frames
frameWidth = video.Width / 2;  % Divide the frame width in half to accommodate the 3D and 2D plots
frameHeight = video.Height;
imshow(zeros(frameHeight, frameWidth, 3), 'Parent', ax1);  % Display a black background
set(ax1, 'Color', [0.1 0.1 0.1]);  % Set the background color of ax1 to dark gray
set(ax1, 'XColor', '#66ff99', 'YColor', '#66ff99');  % Set the axis numbers color to light green

% Plot the 3D embeddings with a fade
scatter3(ax2, embeddings(:, 1), embeddings(:, 2), embeddings(:, 3), 30, labels, 'filled', 'MarkerFaceAlpha', 0.1);
colormap(ax2, magmaColormap);
colorbar(ax2, 'Ticks', unique(labels), 'TickLabels', cellstr(num2str(unique(labels))), 'Color', 'w');
title(ax2, '3D Embeddings');
set(ax2, 'Color', [0.1 0.1 0.1]);  % Set the background color of ax2 to dark gray
set(ax2, 'XColor', '#66ff99', 'YColor', '#66ff99', 'ZColor', '#66ff99');  % Set the axis numbers color to light green
view(ax2, 3);  % Set the 3D view

% Plot the 2D embeddings with a fade
scatter(ax3, tD_embeddings(:, 1), tD_embeddings(:, 2), 30, labels, 'filled', 'MarkerFaceAlpha', 0.1);
colormap(ax3, magmaColormap);
colorbar(ax3, 'Ticks', unique(labels), 'TickLabels', cellstr(num2str(unique(labels))), 'Color', 'w');
title(ax3, '2D Embeddings');
set(ax3, 'Color', [0.1 0.1 0.1]);  % Set the background color of ax3 to dark gray
set(ax3, 'XColor', '#66ff99', 'YColor', '#66ff99');  % Set the axis numbers color to light green

% Add a title text box
t = 'PoseModel: Automated Behavioral State Estimation - model: GAT-50, IPM-SCS';
annotation(fig, 'textbox', [0.1 0.9 0.8 0.1], 'String', t, ...
    'Color', 'w', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 19, 'FontWeight', 'bold');

% Set up GIF parameters
fps = 30;
delayTime = 1 / fps;

% Create text annotations for time, frame number, and state
textAnnotation1 = annotation(fig, 'textbox', [0.035 0.75 0.11 0.05], 'String',...
    '', 'Color', 'w', 'EdgeColor', 'w','FontSize',13, 'fontweight', 'bold');
textAnnotation2 = annotation(fig, 'textbox', [0.035 0.70 0.11 0.05], 'String',...
    '', 'Color', 'w', 'EdgeColor', 'w', 'FontSize',13, 'fontweight', 'bold');
textAnnotation3 = annotation(fig, 'textbox', [0.035 0.65 0.11 0.05], 'String',...
    '', 'Color', 'w', 'EdgeColor', 'w','FontSize',13, 'fontweight', 'bold');

% Capture frames and create GIF
for i = 1:length(labels)
    % Read the video frame
    frame = readFrame(video);
    
    % Display the video frame
    imshow(frame, 'Parent', ax1);
    
    % Remove the previous current state dot in the 3D plot
    if i > 1
        delete(findobj(ax2, 'MarkerEdgeColor', 'r'));  % Delete the previous red dots
    end
    
    % Remove the previous current state dot in the 2D plot
    if i > 1
        delete(findobj(ax3, 'MarkerEdgeColor', 'r'));  % Delete the previous red dots
    end
    
    % Update the current embedding dot in the 3D plot
    hold(ax2, 'on');
    scatter3(ax2, embeddings(i, 1), embeddings(i, 2), embeddings(i, 3), 60, 'w', 'filled', 'MarkerEdgeColor', 'r');
    scatter3(ax2, embeddings(i, 1), embeddings(i, 2), embeddings(i, 3), 90, 'w', 'filled', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);
    hold(ax2, 'off');
    
    % Update the current embedding dot in the 2D plot
    hold(ax3, 'on');
    scatter(ax3, tD_embeddings(i, 1), tD_embeddings(i, 2), 60, 'w', 'filled', 'MarkerEdgeColor', 'r');
    scatter(ax3, tD_embeddings(i, 1), tD_embeddings(i, 2), 90, 'w', 'filled', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);
    hold(ax3, 'off');
    
    % Update the text annotations
    set(textAnnotation1, 'String', sprintf('Time: %.2f s', i/30));
    set(textAnnotation2, 'String', sprintf('Frame: %d', i));
    set(textAnnotation3, 'String', sprintf('State: %s', num2str(labels(i))));
    
    % Capture the current figure as a frame for the GIF
    frame = getframe(fig);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    % Write the frame to the GIF file
    if i == 1
        imwrite(imind, cm, outDir, 'gif', 'Loopcount', inf, 'DelayTime', delayTime);
    else
        imwrite(imind, cm, outDir, 'gif', 'WriteMode', 'append', 'DelayTime', delayTime);
    end
end

disp('GIF file created successfully.');