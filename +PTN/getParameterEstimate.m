function [p] = getParameterEstimate(Q, node, qps, pdt, p)
% Get parameter estimate for the given qps, relative to the belief state
% pdt. A QPS might reference only part of a node's tensor, so update the
% relevant part.
    thetaEps = 1e-12;
    for tryExp=1:10
        try
            dist = PTN.sparse_exp_normcore({pdt}, 1e-14, 'nswp', 49, 'verb', 0);
            break;
        catch ME
            % If extremely negative log values, svd can fail in amen, so retry
            warning('SVD failed in amen_cross, retrying: %s\n', getReport(ME));
        end
    end
    
    dist = dist / sum(dist);

    unStoch = Q.unStoch{node};
    idx = cell(1, Q.arity(node));

    if numel(unStoch) > 0
        [idx{unStoch}] = ind2sub(Q.size{node}(unStoch), qps);
    end

    outShape = Q.size{node}(Q.stoch{node});

    % For each output index
    for k=1:prod(outShape)
        [idx{Q.stoch{node}}] = ind2sub(outShape, k);

        p(idx{:}) = dot(dist, Q.lambda{node}{qps}{k});
              
        if(p(idx{:}) < thetaEps)
            warning('Clamping a small or negative value to epsilon');
            p(idx{:}) = thetaEps;
        end
    end
    
    % Renorm, in rare cases where values got clamped
    idx(Q.stoch{node}) = repmat({':'}, 1, numel(outShape));
    renorm = p(idx{:});
    renorm = sum(renorm(:));
    p(idx{:}) = p(idx{:}) / renorm;
end

