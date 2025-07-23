function saveOutputBranchGraphsManual(matFile)
    % Load the trained network
    data = load(matFile);
    
    if isfield(data, 'dlnet')
        lgraph = layerGraph(data.dlnet);
    else
        error('No dlnet found in %s', matFile);
    end

    % Output layer names
    outputLayerNames = {'out_b1', 'out_b2', 'out_b3', 'out_b4'};
    branchNames = {'Anthocyanin', 'TPC', 'TFC', 'DPPH'};

    for i = 1:numel(outputLayerNames)
        try
            outputName = outputLayerNames{i};
            branchName = branchNames{i};

            % Manually trace relevant layers to each output
            layersToInclude = findLayersToBranch(lgraph, outputName);

            % Create subgraph with only selected layers
            branchGraph = createSubGraph(lgraph, layersToInclude);

            % Plot and save
            figure('Visible', 'off');
            plot(branchGraph);
            title(sprintf('Output Branch: %s', branchName));
            saveas(gcf, sprintf('BranchGraph_%s.png', branchName));
            close;
        catch ME
            warning("Failed to save branch for %s\nReason: %s", outputName, ME.message);
        end
    end
end

function layers = findLayersToBranch(lgraph, outputName)
    % Manually backtrack layers from a given output
    % Define connections backwards from each output
    switch outputName
        case 'out_b1'
            targetLayers = {'reg_head1', 'cbam_mult', 'res5b_relu', 'input'};
        case 'out_b2'
            targetLayers = {'reg_head2', 'cbam_mult_1', 'res5b_relu', 'input'};
        case 'out_b3'
            targetLayers = {'reg_head3', 'cbam_mult_2', 'res5b_relu', 'input'};
        case 'out_b4'
            targetLayers = {'reg_head4', 'cbam_mult_3', 'res5b_relu', 'input'};
        otherwise
            error("Unknown output name: %s", outputName);
    end

    % Match actual layer names
    layers = lgraph.Layers(ismember({lgraph.Layers.Name}, targetLayers));
end

function subgraph = createSubGraph(lgraph, selectedLayers)
    % Extract the minimal LayerGraph containing only selected layers
    layerNames = {selectedLayers.Name};
    connections = lgraph.Connections;

    % Filter only connections among selected layers
    subConnections = connections(ismember(connections.Source, layerNames) & ...
                                  ismember(connections.Destination, layerNames), :);

    % Build new graph
    subgraph = layerGraph();
    for i = 1:numel(selectedLayers)
        subgraph = addLayers(subgraph, selectedLayers(i));
    end
    subgraph = connectLayersFromTable(subgraph, subConnections);
end

function lgraph = connectLayersFromTable(lgraph, connectionTable)
    for i = 1:height(connectionTable)
        lgraph = connectLayers(lgraph, ...
            connectionTable.Source{i}, connectionTable.Destination{i});
    end
end
