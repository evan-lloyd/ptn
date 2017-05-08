function [ phi ] = do_log_eval( net, parms, order )
    % Evaluate (in log-space) the given (TN, parameters). If no evaluation order is given,
    % go in ascending order.
    if nargin < 3
        order = arrayfun(@(x) {x}, 1:net.n, 'UniformOutput', false);
    end
    [~, phi] = PTN.log_tn_eval(net, parms, order);
    phi = phi{1};
end

