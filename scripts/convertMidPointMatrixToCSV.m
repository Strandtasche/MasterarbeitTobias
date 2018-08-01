% created: 2018-07-26
% purpose: convert midpointMatrix .mat file to csv

function[] = convertMidPointMatrixToCSV(midpointMatrix, dest_path)
    disp('starting script')
    
    %If sourcePath ist string: load(sourcepath)
    % wenn variable do whatevs
    data = zeros(19999, 2*207 + 2);
    for i=1:size(midpointMatrix,3)
        if mod(i, 100) == 0
            disp(i)
        end
        data(i, 1) = i;
        numberNonNan = sum(~isnan(midpointMatrix(:,:,i)), 2);
        data(i, 2) = numberNonNan(1);
        for j=1:size(midpointMatrix,2)
           for k=1:size(midpointMatrix,1)
               data(i, 2*j + (k)) = midpointMatrix(k, j, i);
           end
        end
    end
    
    disp("writing header")
    cHeader = ["FrameNr" "NumberMidPoints"]; %dummy header
    for i=1:207
        cHeader(end+1) = strrep("MidPoint_Val_x", "Val", string(i));
        cHeader(end+1) = strrep("MidPoint_Val_y", "Val", string(i));
    end
    %disp(class(cHeader(3)))
    commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commas
    commaHeader = commaHeader(:)';
    %disp(class(commaHeader(1)))
    %disp(class(commaHeader(2)))
    textHeader = cell2mat(commaHeader); %cHeader in text with commas
    %textHeader = strjoin(cHeader, ',');
    
    %write header to file
    fid = fopen(dest_path,'w'); 
    fprintf(fid,'%s\n',textHeader)
    fclose(fid);
    %write data to end of file
    dlmwrite(dest_path, data, '-append');
    
    %csvwrite(dest_path, data)
end