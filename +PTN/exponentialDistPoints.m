function [ pts ] = exponentialDistPoints( n )
% Get a regular list of n points, i.e. one that has a low TT-rank!, that just
% so happen to follow an exponential distribution with \lambda=1.
% This is actually pretty easy with a simple trick: we take a transform
% that would map the uniform distribution onto the desired distribution,
% then just apply it to a list of evenly separated points in [0,1].
% Here we use http://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates
% to generate an exponential with \lambda = 1;
    if n < 2^21
        n = n - 2; % replace each end with "extreme" values
        u = ((1:n) - 0.5) / n;
        % Perturb slightly to improve coverage
        u = u + (rand(1,n) - 0.5) / (2*n);

        pts = [-log(2^-20) -log(u) -log(1 - 2^-20)];
    else
        % Resolution is high, gets close to 0/1 anyway
        u = ((1:n) - 0.5) / n;
        % Perturb slightly to improve coverage
        u = u + (rand(1,n) - 0.5) / (2*n);

        pts = -log(u);
    end
end

