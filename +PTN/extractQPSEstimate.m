function [thisEst] = extractQPSEstimate(Q, node, qps, p)
% Extract the part of a parameter estimate for a single QPS
    unStoch = Q.unStoch{node};
    idx = cell(1, Q.arity(node));

    if numel(unStoch) > 0
        [idx{unStoch}] = ind2sub(Q.size{node}(unStoch), qps);
    end
    
    idx(Q.stoch{node}) = repmat({':'}, 1, numel(Q.stoch{node}));
    thisEst = p(idx{:});
end

