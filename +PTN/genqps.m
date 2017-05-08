function [t, logT] = genqps(shape, res)
% Generate a quantized parameter set with an approach similar to Algorithm 2 of
% "Conditional Distribution Inverse Method in Generating Uniform Random Vectors
% Over a Simplex" (Moeini, Abbasi, and Mahlooji, 2011), converted to tensor
% form.
    n = prod(shape);
    
    % Number of amen sweeps; odd number seems to result in slightly higher
    % accuracy
    nAmenSwp = 49;
    
    if nargin < 2
        res = 3;
    end
    
    if numel(shape) == 1
        x = cell(shape, 1);
        z = cell(shape, 1);
        logX = cell(shape, 1);
    else
        x = cell(shape);
        z = cell(shape);
        logX = cell(shape);
    end
    
    % Make quantics tensor train with each dimension 2 (though others are
    % possible)
    dimSize = 2;
    nPoints = dimSize^res;
    effectiveRes = res;
    baseShape = repmat(dimSize, 1, effectiveRes);
    xShape = repmat(dimSize, 1, effectiveRes*n);
    
    for i=1:n
        % Reshape x to fit these coords in the right chunk; there are
        %   - some number of coords to our LEFT: (i-1)
        %   - then there's us
        %   - then there's some number of coords to our RIGHT: (n - i)
        z{i} = PTN.exponentialDistPoints(nPoints);
        tt = reshape(tt_tensor(z{i}), baseShape, 1e-14);
        x{i} = add_non_essential_dims(tt, xShape, ((i-1)*effectiveRes)+1:((i)*effectiveRes));
    end
    
    % If requested, use amen_cross to get pointwise logarithm of the QPS
    if nargout > 1
        for i=1:n
            tt = reshape(tt_tensor(log(z{i})), baseShape, 1e-14);
            logX{i} = add_non_essential_dims(tt, xShape, ((i-1)*effectiveRes)+1:((i)*effectiveRes));
        end
    end
    
    norm = PTN.tt_sum(x{:});
    if nargout > 1
        logNorm = amen_cross({norm}, @log, 1e-14, 'nswp', nAmenSwp, 'verb', 0);
    end
    
    if numel(shape) == 1
        t = cell(shape, 1);
    else
        t = cell(shape);
    end
    
    % Normalize
    for i=1:prod(shape)
        t{i} = amen_cross({x{i}, norm}, @(y) y(:,1) ./ y(:,2), 1e-14, 'nswp', nAmenSwp, 'verb', 0);
    end
    
    if nargout > 1
        if numel(shape) == 1
            logT = cell(shape, 1);
        else
            logT = cell(shape);
        end
        for i=1:prod(shape)
            logT{i} = logX{i} - logNorm;
        end
    end
end