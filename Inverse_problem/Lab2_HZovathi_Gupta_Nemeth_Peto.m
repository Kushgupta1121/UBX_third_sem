%% Students:
%  Levente Pető
%  Bendegúz H.Zováthi
%  Kush Gupta
%  Sebestyén Németh

% Clean the workspace and load the data

close all, clear all;
% Loading the Data 1 and Data 2
Data1 = load('DataOne.mat');
Data2 = load('DataTwo.mat');

T = 0.2; % Threshold
alpha = 0.045; %0.45 with smaller alpha it will converge slower
num_iter = 3000; %300
mu = 0.1;

% testing for different alpha values, smaller value will make 
% the convergence slower. Values > 1 will diverge
data=Data1; % Passing Data1 to the calculate function

calculate(data, T, 0.5, num_iter, mu)
calculate(data, T, 0.05, num_iter, mu)
calculate(data, T, 0.005, num_iter, mu)
calculate(data, T, 1e-5, num_iter, mu)

% Funtion to do the Image Restoration
function calculate(Data, T, alpha, num_iter, mu)

    [h, w] = size(Data.Data);
    
    mu_p = mu / alpha;
    mse = zeros(num_iter, 1);
    
    % Difference matrix
    D_col = [0  0 0;
             0 -1 1;
             0  0 0];
    D_row = [0  0 0;
             0 -1 0;
             0  1 0];
    
    %Calculating FFT of our Observed Image
    I = MyFFT2(Data.Data);
    
    % Calculating the FFT of the Difference matrix row and column wise
    D_col_fft = MyFFT2RI(D_col, h);
    D_row_fft = MyFFT2RI(D_row, w);
    
    %Calculating the interpixel difference row and column wise.
    delta_col = MyIFFT2(D_col_fft .* I);
    delta_row = MyIFFT2(D_row_fft .* I); 
    
    %Updating the auxiliary variable a row and column wise.
    a_col = (1 - 2 * alpha * min(1, T ./ abs(delta_col))) .* delta_col;
    a_row = (1 - 2 * alpha * min(1, T ./ abs(delta_row))) .* delta_row;
    
    % Calculating the Denominator parts of the equation to be minimized
    H_square_fft = abs(MyFFT2RI(Data.IR, h)) .^ 2;
    D_square_fft = abs(D_col_fft) .^ 2 + abs(D_row_fft) .^ 2;

    % Calculating the numerator parts of the equation to be minimized
    H_conj_fft = conj(MyFFT2RI(Data.IR, h));
    D_col_conj_fft = conj(D_col_fft);
    D_row_conj_fft = conj(D_row_fft);
    
    a_col_fft = MyFFT2(a_col);
    a_row_fft = MyFFT2(a_row);

    x = Data.Data;
    prev_x = zeros(size(Data.Data));
    
    for k=1:num_iter % 
        x_denom = H_square_fft + mu_p * D_square_fft;
        x_num = H_conj_fft .* I + ...
            mu_p * D_col_conj_fft .* a_col_fft + ...
            mu_p * D_row_conj_fft .* a_row_fft;
        
        % the final equation to be minimised
        x_fft = x_num ./ x_denom;
        
        %
        delta_col = MyIFFT2(D_col_fft .* x_fft);
        delta_row = MyIFFT2(D_row_fft .* x_fft);
        
        % Updating a
        a_col = (1 - 2 * alpha * min(1, T ./ abs(delta_col))) .* delta_col;
        a_row = (1 - 2 * alpha * min(1, T ./ abs(delta_row))) .* delta_row;
    
        a_col_fft = MyFFT2(a_col);
        a_row_fft = MyFFT2(a_row);
        
        prev_x = x;
        x = MyIFFT2(x_fft);
        mse(k) = d_2(x, Data.TrueImage);
        if d_2(x, prev_x) < 1e-14
            break
        end
    end
    
    figure("Name",sprintf("alpha = %.3f", alpha));
    
    subplot(231)
    imagesc(Data.TrueImage);
    title('True Image')
    colormap('gray'); colorbar; axis('square')
    
    subplot(232)
    imagesc(Data.Data);
    title('Observed Image')
    colormap('gray'); colorbar; axis('square')

    subplot(233)
    imagesc(x);
    title('Resconstructed Image')
    colormap('gray'); colorbar; axis('square')

    subplot(234)
    Log_img = 20*log10(abs(x_fft));
    imagesc(Log_img);
    title('FFT of Image - Log space')
    colormap('gray'); colorbar; axis('square')

    subplot(235)
    plot(mse(mse > 0))
    title('d_2 error')
end

% Function to compute the error, distance between the reconstructed image
% and the true image
function d = d_2(rec, true)
    d = sum((rec - true).^2, "all") / sum(true.^2, "all");
end