function [ phi ] = do_log_subtn_eval( net, parms, outIdx, node, nodeIdx, order )
    % Evaluate (in log-space) the sub-network of the given (TN, parameters) at the output multi-index outIdx,
    % with the node "node" removed, and its neighbors flattened to be consistent with the indices
    % given in nodeIdx. If no order given, evaluate in ascending order.
    if nargin < 6
        order = arrayfun(@(x) {x}, 1:net.n-1, 'UniformOutput', false);
    end
    
    % Extract subnet that excludes node and selects output index
    
    % First, flatten/remove output dimensions
    for i=1:net.n
        % Removing it anyway, don't bother
        if i == node
            continue;
        end
        
        parmIdx = repmat({':'}, 1, net.arity(i));
        outWays = net.outcomeWays(net.nodeOutcomes{i});
        parmIdx(outWays) = outIdx(net.nodeOutcomes{i});
        net.size{i}(outWays) = [];
        net.arity(i) = numel(net.size{i});
        
        parms{i} = squeeze(parms{i}(parmIdx{:}));
        
        if numel(net.size{i}) == 1
            parms{i} = reshape(parms{i}, net.size{i}(1), 1);
        end
    end
    
    % NB: no need to update edges at this point, since output edges are
    % assumed to come after any contraction edges
    
    % Flatten contractions with target node, remove its edges, and update
    % remaining edge's node indices
    newEdges = [];
    removedWays = cell(1, net.n);
    targetWays = cell(1, net.n);
    % Remove edges with target node
    for e=net.edge'
        if e(1) == node || e(3) == node
            if e(1) == node
                affectedNode = e(3);
                removedWays{affectedNode}(end+1) = e(4);
                targetWays{affectedNode}(end+1) = nodeIdx{e(2)};
            end
        else
            % Update unaffected edges with new node indices
            if e(1) > node
                e(1) = e(1) - 1;
            end
            if e(3) > node
                e(3) = e(3) - 1;
            end
            newEdges(end+1, :) = e;
        end
    end
    
    % Update ways in edges to account for those removed
    for i=1:size(newEdges,1)
        % Indices in edge possibly changed by update, get correct
        % removedWays by adding one to index if so
        newEdges(i,2) = update(newEdges(i,2), removedWays{newEdges(i,1) + (newEdges(i,1) >= node)});
        newEdges(i,4) = update(newEdges(i,4), removedWays{newEdges(i,3) + (newEdges(i,3) >= node)});
    end
    net.edge = newEdges;
    
    % Select appropriate sub-tensors of any flattened nodes
    for i=1:net.n
        if i == node
            continue;
        end
        
        if ~isempty(targetWays{i})
            parmIdx = repmat({':'}, 1, net.arity(i));
            parmIdx{removedWays{i}} = targetWays{i};
            net.size{i}(removedWays{i}) = [];
            net.arity(i) = numel(net.size{i});
             
            parms{i} = squeeze(parms{i}(parmIdx{:}));
            
            if numel(net.size{i}) == 1
                parms{i} = reshape(parms{i}, net.size{i}(1), 1);
            end
        end
    end
    
    
    % Evaluate the sub-network
    parms(node) = [];
    net.n = net.n - 1;
    net.arity(node) = [];
    net.size(node) = [];
    
    [~, phi] = PTN.log_tn_eval(net, parms, order);
    phi = phi{1};
end

function [f] = update(f, x)
    if ~isempty(x)
        f = f - arrayfun(@(y) sum(y > x), f);
    end
end

