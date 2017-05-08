function [dist, targetParms] = generate_ptn_dist(Q, method)
    % Generate a distribution matching the output shape of the PTN for Q,
    % using the given method, which should be one in the enumeration of
    % randomPTNTargetTypes().
    
    switch method
        % Select a random quantization for each parameter, evaluate the
        % network with those parameters to generate target.
        case 'randIndex'
            targetParms = cell(1, Q.n);
            for i=1:Q.n
                if ~isempty(Q.fixedParameters{i})
                    targetParms{i} = log(Q.fixedParameters{i});
                    continue;
                end
                unStoch = Q.unStoch{i};
                if Q.arity(i) > 1
                    targetParms{i} = zeros(Q.size{i});
                else
                    targetParms{i} = zeros(Q.size{i}, 1);
                end
                
                % Select quantization from each stochastic block
                if numel(unStoch) > 0
                    qpsIdx = repmat({':'},1, Q.arity(i));
                    for j=1:prod(Q.size{i}(unStoch))
                        
                        [qpsIdx{unStoch}] = ind2sub(Q.size{i}(unStoch), j);
                        idx = num2cell(round(rand(1, prod(Q.size{i}(Q.stoch{i}))*Q.res)) + 1);
                        
                        if numel(unStoch) > 1
                            t = arrayfun(@(x) Q.logLambda{i}{j}{x}(idx{:}), ...
                                                 1:prod(Q.size{i}(Q.stoch{i})));
                            if numel(Q.stoch{i}) > 1
                                targetParms{i}(qpsIdx{:}) = reshape(t, Q.size{i}(Q.stoch{i}));
                            else
                                targetParms{i}(qpsIdx{:}) = reshape(t, Q.size{i}(Q.stoch{i}), 1);
                            end
                        else
                            targetParms{i}(qpsIdx{:}) = ...
                                arrayfun(@(x) Q.logLambda{i}{j}{x}(idx{:}), ...
                                                 1:prod(Q.size{i}(Q.stoch{i})));
                        end
                    end
                else % Fully stochastic, just get a random JPT
                    idx = num2cell(round(rand(1, prod(Q.size{i})*Q.res)) + 1);
                    if Q.arity(i) > 1
                        targetParms{i} = reshape(arrayfun(@(x) Q.logLambda{i}{1}{x}(idx{:}), ...
                                                 1:prod(Q.size{i})), Q.size{i});
                    else
                        targetParms{i} = arrayfun(@(x) Q.logLambda{i}{1}{x}(idx{:}), ...
                                                 1:prod(Q.size{i}))';
                    end
                end
            end
            [~, logT] = PTN.log_tn_eval(Q, targetParms, arrayfun(@(x) {x}, ...
                                    1:numel(Q.arity), 'UniformOutput', false));
            dist = exp(logT{1});

        % Generate a random tensor of the appropriate shape and
        % stochasticity for each parameter, evaluate the network with
        % those parameters to generate target.
        case 'randSimplex'
            targetParms = cell(1, Q.n);
            for i=1:Q.n
                if ~isempty(Q.fixedParameters{i})
                    targetParms{i} = log(Q.fixedParameters{i});
                    continue;
                end
                unStoch = setdiff(1:Q.arity(i), Q.stoch{i});
                targetParms{i} = zeros(Q.size{i});

                idx = repmat({':'},1, Q.arity(i));

                % Generate each stochastic block independently
                if numel(unStoch) > 0
                    for j=1:prod(Q.size{i}(unStoch))
                        [idx{unStoch}] = ind2sub(Q.size{i}(unStoch), j);
                        targetParms{i}(idx{:}) = log(PTN.test.randSimplex(Q.size{i}(Q.stoch{i})));
                    end
                else % Fully stochastic, just get a random JPT
                    targetParms{i} = log(PTN.test.randSimplex(Q.size{i}));
                end
            end
            [~, logT] = PTN.log_tn_eval(Q, targetParms, arrayfun(@(x) {x}, ...
                                    1:numel(Q.arity), 'UniformOutput', false));
            dist = exp(logT{1});

        % Generate a random JPT of the appropriate shape as the target.
        case 'randJPT'
            dist = PTN.test.randSimplex(Q.outcomeShape);
            targetParms = {};
        otherwise
            error('Unrecognized generation method "%s"', method);
    end
    targetParms = cellfun(@(x) exp(x), targetParms, 'UniformOutput', false);
end