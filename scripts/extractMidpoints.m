% created: 2018-07-26
% purpose: convert midpointMatrix .mat file to csv

function[] = extractMidpoints(midpointMatrix, midpointToGtMapping, dest_path)
    if nargin==0
        midpointMatrix=evalin('base','midpointMatrix');
    end
    if nargin<2
        midpointToGtMapping=evalin('base','midpointToGtMapping');
    end
    
    numberColumns = 2*4427
    data = NaN(19999, numberColumns);
    for i=1:size(midpointMatrix, 3)
        numberNonNan = sum(~isnan(midpointMatrix(:,:,i)), 2);
        k = 1;
        while ~isnan(midpointToGtMapping(k, i)) && (k < 207)
            data(i, 2*(midpointToGtMapping(k, i)-1) +1) = midpointMatrix(1, k, i);
            data(i, 2*(midpointToGtMapping(k, i)-1) + 2) = midpointMatrix(2, k, i);
            k = k + 1;
        end
    end
    
    %postprocessing
    maxColumn = max(sum(~isnan(data(:,:))));
    
    dataFinal = NaN(maxColumn, numberColumns);
    valueCounter = 0
    for c=1:numberColumns
        no = sum(~isnan(data(:,c)));
        if no==0
            valueCounter = valueCounter + 1;
        end
        dataFinal(1:no, c) = data(~isnan(data(:, c)), c);
    end
    
    disp(["Number of Colums with only NaN Values: ", num2str(valueCounter)])
    
%     disp(dataFinal(:, 1:5))
%     f1 = figure;
%     plot(dataFinal(:, 11), dataFinal(:,12))
%     f2 = figure;
%     plot(dataFinal(:, 13), dataFinal(:,14))
%     f3 = figure;
%     plot(dataFinal(:, 15), dataFinal(:,16))
%     min(sum(~isnan(dataFinal(:,
%     return
%     
    
    disp("writing header")
    cHeader = {};
    for i=1:4427
        cHeader(end+1) = {strrep('TrackID_Val_X', 'Val', char(string(i)))};
        cHeader(end+1) = {strrep('TrackID_Val_Y', 'Val', char(string(i)))};
    end
    
    commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commas
    commaHeader = commaHeader(:)';
    
    textHeader = cell2mat(commaHeader);
    
    fid = fopen(dest_path,'w'); 
    fprintf(fid,'%s\n',textHeader)
    fclose(fid);
    %write data to end of file
    dlmwrite(dest_path, dataFinal, '-append');
    
end