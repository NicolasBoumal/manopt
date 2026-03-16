%This is the code for the denoising experiment in the paper 
%
% 'Riemannian Optimization on Relaxed Indicator Matrix Manifold' 
%
% which allows you to compare the time required for denoising via the RIM manifold and the doubly stochastic manifold.

% Step 1: Load and preprocess the image
rng(1);
img = imread('.\yjh.png');
img = im2double(img); % Normalize pixel values to [0, 1]

% Step 2: Add noise to each channel of the image
[rows, cols, channels] = size(img);
sigma = 0.3;

% Add noise to each channel
A = zeros(size(img));
for ch = 1:channels
    A(:, :, ch) = max(img(:, :, ch) + sigma * randn(rows, cols), 0.01);
end

% Step 3: Adjust row and column sum constraints for each channel
row_sums = cell(1, channels);
col_sums = cell(1, channels);
for ch = 1:channels
    row_sums{ch} = sum(A(:, :, ch), 2); % Row sums for channel `ch`
    col_sums{ch} = sum(A(:, :, ch), 1)'; % Column sums for channel `ch`
end

% Step 4: Define optimization problem for each channel
lambda = .3; % Regularization coefficient

% Define difference operators
Dx = @(X) X(:, [2:end, end]) - X; % Horizontal difference
Dy = @(X) X([2:end, end], :) - X; % Vertical difference

% Define TV regularization term and gradient
TV = @(X) sum(sum(abs(Dx(X)) + abs(Dy(X)))); % Anisotropic TV
gradTV = @(X) sign(Dx(X)) - sign([zeros(size(X,1),1), Dx(X(:,1:end-1))]) + ...
             sign(Dy(X)) - sign([zeros(1,size(X,2)); Dy(X(1:end-1,:))]);

% Initialize the reconstructed image
X_reconstructed = zeros(size(img));

% Loop through each channel
tic
for ch = 1:channels
    % Define manifold for current channel
    manifold = relaxdindicatormatrixfactory(rows, cols, row_sums{ch}, col_sums{ch}, col_sums{ch});
    %manifold = multinomialdoublystochasticgeneralfactory(rows, cols, row_sums{ch}, col_sums{ch});
    
    % Define optimization problem for current channel
    problem.M = manifold;
    problem.cost = @(X) 0.5 * norm(X - A(:, :, ch), 'fro')^2 + lambda * TV(X);
    problem.egrad = @(X) (X - A(:, :, ch)) - lambda * gradTV(X);
    
    % Solve optimization problem
    [X_reconstructed(:, :, ch), ~, info, ~] = steepestdescent(problem);
end
time_cost=toc;
value=problem.cost(X_reconstructed(:, :, 1))+problem.cost(X_reconstructed(:, :, 2))+problem.cost(X_reconstructed(:, :, 3));
% Step 5: Visualize the results
figure;
subplot(1, 3, 1); imshow(img); title('Original Image');
subplot(1, 3, 2); imshow(A); title('Noisy Image');
subplot(1, 3, 3); imshow(X_reconstructed); title('Reconstructed Image');

