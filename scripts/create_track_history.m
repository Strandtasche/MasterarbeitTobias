% Usage: 
% add nessesary things to path -addpath(genpath('matlab') ?
% Parameters:
%   source_path: absolute path to csv-file of the observations to create gt
%   from
%   dest_path: path to directory where the ground_truth data shall be saved
%   to
% >> reorder_centroids
function[] = create_track_history(source_path, dest_path)
    disp('starting script')

    %Set magic Parameters, so the tracking works well
    allParam=getDefaultParam([85;0]);
    allParam.score.looseStep=35;
    allParam.association.useOrientation=false;
    allParam.initial.PositionCov=50*eye(2);
    allParam.meas.PositionCov=50*eye(2);
    allParam.association.tryToUseMex=false;
    %allParam.association.distanceMetricPos='Euclidean';
    allParam.general.rotateBy=pi;
    disp('parameters set!')

    %files = dir('*_data.csv');
    files = dir(source_path)
    
    i = 1;
    for file = files'
        
        disp(strcat('Processing : ', file.name))
        trackHistory=tracksortAlgorithm([0,2320;0,1728],1300,1450,allParam,file.name);
        
        trackHistory_nothingDeleted = trackHistory;
        %These numbers in this command should be fitted for the input data, but
        %same camera setup should be able to use the same ones.
        THRESH_MINNANS = 4;
        THRESH_BACKWARDS = 3;
        %first argument of testSuite 
        trackHistory = preprocessTrackHistory(trackHistory, true, THRESH_MINNANS, THRESH_BACKWARDS);
        dest_path = strcat(dest_path,"/");
        savingFileNameTrackHistory = strcat(file.name(1:end-numel('.csv')),'_trackHistory.mat');
        savingPathTrackHistory = strcat(dest_path, savingFileNameTrackHistory);
        
        %save trackhistory to file
        save(savingPathTrackHistory, "trackHistory");

        i = i + 1;
        assembledRawMeasurements = transpose(trackHistory(1).RawMeasurements);
        for it = 2:size(trackHistory,2)
            currentRawMeasurements = transpose(trackHistory(it).RawMeasurements);

            rowsA = size(assembledRawMeasurements,1);
            rowsB = size(currentRawMeasurements,1);
            if rowsA ~= rowsB
                if rowsA > rowsB
                    currentRawMeasurements = vertcat(currentRawMeasurements, NaN(rowsA-rowsB, size(currentRawMeasurements,2)));
                else
                    assembledRawMeasurements = vertcat(assembledRawMeasurements, NaN(rowsB-rowsA, size(assembledRawMeasurements,2)));
                end
            end
            assembledRawMeasurements = horzcat(assembledRawMeasurements,currentRawMeasurements);
        end
        

        % write header
        num_tracks = size(trackHistory,2);
        header = {};

        for i=1:num_tracks
            header(end+1) = {strcat(strcat('TrackID_',num2str(i)),'_X')};
            header(end+1) = {strcat(strcat('TrackID_',num2str(i)),'_Y')};
        end

        % save it in csv file
        data_table = array2table(assembledRawMeasurements, 'VariableNames', header);
        csvFileNameNotShifted = savingFileNameTrackHistory(1:strfind(savingFileNameTrackHistory,'_trackHistory.mat')-1);
        appendix = strcat('_trackLabels_NaN',num2str(THRESH_MINNANS));
        appendix = strcat(appendix,'_BW');
        appendix = strcat(appendix,num2str(THRESH_BACKWARDS));
        appendix = strcat(appendix,'_NotShifted.csv');
        csvFileNameNotShifted = strcat(csvFileNameNotShifted, appendix);
        writetable(data_table, strcat(dest_path, csvFileNameNotShifted));

        % shift observations accordingly
        for currentTrackID=1:num_tracks
            startTime = trackHistory(currentTrackID).StartTime;

            if startTime > 1
                trackObservations = assembledRawMeasurements(:, 2*currentTrackID-1:2*currentTrackID);
                % shift vector by startTime - 1
                shift = NaN(startTime-1, 2);
                trackObservations = vertcat(shift, trackObservations);
                % now trim it
                trackObservations = trackObservations(1:end-startTime + 1,:);
                % replace vector accordingly
                assembledRawMeasurements(:, 2*currentTrackID-1:2*currentTrackID) = trackObservations;
            end
        end

        % save it in csv file
        csvFileNameShifted = csvFileNameNotShifted(1:strfind(csvFileNameNotShifted, '_NotShifted.csv')-1);
        csvFileNameShifted = strcat(csvFileNameShifted, '_Shifted.csv');
        data_table = array2table(assembledRawMeasurements, 'VariableNames', header);
        writetable(data_table, strcat(dest_path, csvFileNameShifted));
        
        % now save the trackhistory where nothing was deleted for debug
        % purposes
        
        trackHistory = trackHistory_nothingDeleted;
        assembledRawMeasurements = transpose(trackHistory(1).RawMeasurements);
        for it = 2:size(trackHistory,2)
            currentRawMeasurements = transpose(trackHistory(it).RawMeasurements);

            rowsA = size(assembledRawMeasurements,1);
            rowsB = size(currentRawMeasurements,1);
            if rowsA ~= rowsB
                if rowsA > rowsB
                    currentRawMeasurements = vertcat(currentRawMeasurements, NaN(rowsA-rowsB, size(currentRawMeasurements,2)));
                else
                    assembledRawMeasurements = vertcat(assembledRawMeasurements, NaN(rowsB-rowsA, size(assembledRawMeasurements,2)));
                end
            end
            assembledRawMeasurements = horzcat(assembledRawMeasurements,currentRawMeasurements);
        end
        
        
        % write header
        num_tracks = size(trackHistory,2);
        header = {};

        for i=1:num_tracks
            header(end+1) = {strcat(strcat('TrackID_',num2str(i)),'_X')};
            header(end+1) = {strcat(strcat('TrackID_',num2str(i)),'_Y')};
        end

        % save it in csv file
        data_table = array2table(assembledRawMeasurements, 'VariableNames', header);
        csvFileNothingDeleted = strcat(file.name(1:end-numel('.csv')),'_trackHistory_NothingDeleted.csv');
        writetable(data_table, strcat(dest_path, csvFileNothingDeleted));



    end

    disp('finished')
end