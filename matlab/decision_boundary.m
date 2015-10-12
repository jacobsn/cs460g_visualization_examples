
%
% Example code that shows how to plot the function value and a decision
% boundary for a simple logistic regression model using python
%
% Author: Nathan Jacobs
%

% define the function
f = @(vals) 1./(1+exp(-[0, 1, -.5,.5]*vals));

% define the domain
x_min = -5; x_max = 5;
y_min = -5; y_max = 5;

[x, y] = meshgrid(linspace(x_min,x_max,200), linspace(y_min,y_max,200));

% evaluate
z = f([ones(1,numel(x)); x(:)'; y(:)'; y(:)'.^2]); % lets pretend that this is the result of your kNN
z = reshape(z, size(x));

% plot
figure(1); clf; 
imagesc(z, 'XData', [x_min,x_max], 'YData', [y_min,y_max])
hold on
contour(x,y,z, .5, 'k', 'LineWidth', 2)
hold off
colormap(jet(256))
axis xy, xlabel('X'), ylabel('Y')
colorbar