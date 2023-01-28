%% Students:
%  Levente Pető
%  Bendegúz H.Zováthi
%  Kush Gupta
%  Sebestyén Németh

% Clean the workspace and load the data

close all, clear all
Data1 = load('DataOne.mat');
Data2 = load('DataTwo.mat');

% Display the images
figure("name", "Original Data"), clf
subplot(231)
imagesc(Data1.TrueImage);
colormap('gray');
colorbar;
axis('square','off')
title('True image 1')

subplot(232)
imagesc(Data1.Data);
colormap('gray');
colorbar;
axis('square','off')
title('Data 1')

subplot(233)
imagesc(Data1.IR);
colormap('gray');
colorbar;
axis('square','off')
title('IR 1')

subplot(234)
imagesc(Data2.TrueImage);
colormap('gray');
colorbar;
axis('square','off')
title('True image 2')

subplot(235)
imagesc(Data2.Data);
colormap('gray');
colorbar;
axis('square','off')
title('Data 2')

subplot(236)
imagesc(Data2.IR);
colormap('gray');
colorbar;
axis('square','off')
title('IR 2')

%% Frequency domain

frequency_image(Data1, "Image 1");
frequency_image(Data2, "Image 2");

%% Impulse responses

transfer_function(Data1, "Image 1");
transfer_function(Data2, "Image 2");

%% IFFT

cal_ifft(Data1)
cal_ifft(Data2)

%% Distances

get_distances(Data1)
get_distances(Data2)

%% Functions

function frequency_image(Data, name)
    axis_v = linspace(-0.5, 0.5, 256);

    fft_data = MyFFT2(Data.Data);
    fft_true = MyFFT2(Data.TrueImage);
    figure("name", name + " true");

    magnitude = abs(fft_true);
    phase = atan2(imag(fft_true), real(fft_true));
    
    subplot(131)
    imagesc(axis_v, axis_v, magnitude)
    colormap('gray');
    colorbar;
    axis('square')
    title("Magnitude")
    
    subplot(132)
    imagesc(axis_v, axis_v, log(magnitude))
    colormap('gray');
    colorbar;
    axis('square')
    title("Magnitude log")
    
    subplot(133)
    imagesc(axis_v, axis_v, phase)
    colormap('gray');
    colorbar;
    axis('square')
    title("Phase")

    sgtitle(name + " true")

    figure("name", name + " observed");

    magnitude = abs(fft_data);
    phase = atan2(imag(fft_data), real(fft_data));
    
    subplot(131)
    imagesc(axis_v, axis_v, magnitude)
    colormap('gray');
    colorbar;
    axis('square')
    title("Magnitude")
    
    subplot(132)
    imagesc(axis_v, axis_v, log(magnitude))
    colormap('gray');
    colorbar;
    axis('square')
    title("Magnitude log")
    
    subplot(133)
    imagesc(axis_v, axis_v, phase)
    colormap('gray');
    colorbar;
    axis('square')
    title("Phase")

    sgtitle(name + " data")
end

function transfer_function(Data, name)
    
    axis_v = linspace(-0.5, 0.5, 256);
    
    transfer = abs(MyFFT2RI(Data.IR, 256));
    figure("name", "Impulse responses");
    
    subplot(221)
    imagesc(axis_v, axis_v,Data.IR)
    colormap('gray');
    colorbar;
    axis('square')
    title("IR")

    subplot(222)
    imagesc(axis_v, axis_v, transfer)
    colormap('gray');
    colorbar;
    axis('square')
    title("Transfer function")

    subplot(223)
    plot(axis_v, transfer(128,:))
    title("Slice at 0")

    ax = subplot(224);
    mesh(axis_v, axis_v, transfer)
    colormap(ax, "jet");
    title("3D filter")

    sgtitle(name + " IR")
end

function [x, x_hat] = deconv(obs, IR, mu)
    h = MyFFT2RI(IR, 256);

    d1 = [[0  0 0]
          [0 -1 1]
          [0  0 0]];
    d2 = [[0  0 0]
          [0 -1 0]
          [0  1 0]];

    d = MyFFT2RI(d1, 256) + MyFFT2RI(d2, 256);
    g = conj(h) ./ (abs(h) .^ 2 + mu * abs(d) .^ 2);
    y = MyFFT2(obs);
    x_hat = g .* y;
    x = MyIFFT2(x_hat);
end

function cal_ifft(Data)
mus = [0.01, 0.1, 1, 10];
figure("name", "Spatial domain")
subplot(ceil((length(mus)+1) / 3), 3, 1)
imagesc(Data.TrueImage)
colormap('gray');
colorbar;
axis('square')
title('True image')

subplot(ceil((length(mus)+1) / 3), 3, 2)
imagesc(Data.Data)
colormap('gray');
colorbar;
axis('square')
title('observed image')

for i = 1:length(mus)
    subplot(ceil((length(mus)+1) / 3), 3, i+2)
    [rec, ~] = deconv(Data.Data, Data.IR, mus(i));
    imagesc(rec)
    colormap('gray');
    colorbar;
    axis('square')
    title(sprintf("mu=%2.2f", mus(i)))
end

figure("name", "Frequency domain")
subplot(ceil((length(mus)+1) / 3), 3, 1)
imagesc(log(abs(MyFFT2(Data.TrueImage))))
colormap('gray');
colorbar;
axis('square')
title('True image')

subplot(ceil((length(mus)+1) / 3), 3, 2)
imagesc(log(abs(MyFFT2(Data.Data))))
colormap('gray');
colorbar;
axis('square')
title('observed image')

for i = 1:length(mus)
    subplot(ceil((length(mus)+1) / 3), 3, i+2)
    [~, rec_hat] = deconv(Data.Data, Data.IR, mus(i));
    imagesc(log(abs(rec_hat)))
    colormap('gray');
    colorbar;
    axis('square')
    title(sprintf("mu=%2.2f", mus(i)))
end
end

function d = d_1(rec, true)
    d = sum(abs(rec - true), "all") / sum(abs(true), "all");
end

function d = d_2(rec, true)
    d = sum((rec - true).^2, "all") / sum(true.^2, "all");
end

function d = d_inf(rec, true)
    d = max(abs(rec - true),[],'all') / max(abs(true),[],'all');
end

function get_distances(Data)
    logmus = -5:10;
    d_1_values = zeros(length(logmus), 1);
    d_2_values = zeros(length(logmus), 1);
    d_inf_values = zeros(length(logmus), 1);
    
    for i = 1:length(logmus)
        [rec, ~] = deconv(Data.Data, Data.IR, 10^logmus(i));
        d_1_values(i) = d_1(rec, Data.TrueImage);
        d_2_values(i) = d_2(rec, Data.TrueImage);
        d_inf_values(i) = d_inf(rec, Data.TrueImage);
    end
    
    figure("name", "Distances for different mu values")
    subplot(131)
    hold on
    plot(logmus, d_1_values)
    [~, min_idx] = min(d_1_values);
    [max_v, ~] = max(d_1_values);
    plot([logmus(min_idx), logmus(min_idx)], [0,max_v])
    title(sprintf("mu = %2.2f", 10^logmus(min_idx)))
    
    subplot(132)
    hold on
    plot(logmus, d_2_values)
    [~, min_idx] = min(d_2_values);
    [max_v, ~] = max(d_2_values);
    plot([logmus(min_idx), logmus(min_idx)], [0,max_v])
    title(sprintf("mu = %2.2f", 10^logmus(min_idx)))
    
    subplot(133)
    hold on
    plot(logmus, d_inf_values)
    [~, min_idx] = min(d_inf_values);
    [max_v, ~] = max(d_inf_values);
    plot([logmus(min_idx), logmus(min_idx)], [0,max_v])
    title(sprintf("mu = %2.2f", 10^logmus(min_idx)))
end

