clear all;
clc;

func = @(x1,x2) 100*(x1^2-x2)^2+(x1-1)^2;

gradient = @(x1,x2) [400*x1^3 -400*x1*x2 + 2*x1-2; ...
    200*(x2-x1^2)];

hessian = @(x1,x2) [1200*x1^2-400*x2+2, -400*x1; ...
    -400*x1, 200];

% func = @(x1,x2) 10*(x1^2-x2)^2+(x1-1)^2;
%
% gradient = @(x1,x2) [40*x1^3-40*x1*x2+2*x1-2; ...
%     20*(x2-x1^2)];
%
% hessian = @(x1,x2) [120*x1^2-40*x2+2, -40*x1; ...
%     -40*x1, 20];

x = zeros(2,1); % inital guess of x
c = 0.5;
k = 0;
error = []; X = []; kappa = [];
while 1
    g = gradient(x(1),x(2));
    % Stopping Criteria
    if norm(g,2) < 0.001
        break;
    end
    % Armijo condition
    alpha = 1;
    x_new = x - alpha*g;
    while (func(x_new(1),x_new(2)) > func(x(1),x(2)) - c*alpha*(g'*g))
        alpha = 0.5 * alpha;
        x_new = x - alpha*g;
    end
    % update
    x = x_new;
    k = k+1;
    % save data for visualiztion
    X = [X, x];
    error = [error;func(x(1),x(2))];
    kappa = [kappa;cond(hessian(x(1),x(2)),2)];
end

x
k
error(end)

figure(1);
plot(log(error));
title('ln(error) for each iteration');
ylabel('ln(error)');
xlabel('iter');
grid on;

figure(2);
plot(kappa);
title('Condition Number \kappa for each Iteration');
ylabel('\kappa');
xlabel('iter');
grid on;

figure(3);
[X1,X2]=meshgrid(0:0.01:1.5);
Z = 100*(X1.^2-X2).^2 + (X1-ones(size(X1))).^2;
% Z=100*(Y-X.^2).^2+(ones(size(X))-X).^2;
contour(X1,X2,Z,[0:1:100]);
grid on;
hold on;
plot(X(1,:),X(2,:),'b*');
plot(X(1,:),X(2,:),'r-', 'LineWidth', 1);
ylabel('x_1');
xlabel('x_2');
title('Contour Curve of Rosenbrock Function');

