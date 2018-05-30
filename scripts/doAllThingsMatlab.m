% Usage: 
% 1. add nessesary things to path -addpath(genpath('matlab') ?
% 2. navigate to folder with data you want to process
% 3. optional: change files variable to fit your input csvs.
% >> reorder_centroids

clear all
disp('starting script')

%Set magic Parameters, so the tracking works well
allParam=getDefaultParam([30;0]);
allParam.score.looseStep=30;
allParam.association.useOrientation=false;
allParam.initial.PositionCov=4000*eye(2);
allParam.meas.PositionCov=4000*eye(2);
allParam.association.tryToUseMex=false;
disp('parameters set!')

%files = dir('*_data.csv');
files = dir('Trackhistory_extracted_blobs_RawMeas_NANs.csv');
i = 1;
for file = files'
    trackHistory=tracksortAlgorithm([0,2450;0,1750],1300,1450,allParam,file.name);
    %These numbers in this command should be fitted for the input data, but
    %same camera setup should be able to use the same ones.
    saving = strcat("Trackhistory_", file.name(1:end-4));
    savingFile = strcat(saving, ".mat");
    %save trackhistory to file
    save(savingFile, "trackHistory");
    i = i + 1;
    A = transpose(trackHistory(1).RawMeasurements);
    for it = 2:size(trackHistory,2)
        B = transpose(trackHistory(it).RawMeasurements);
        
        rowsA = size(A,1)
        rowsB = size(B,1)
        if rowsA ~= rowsB
            if rowsA > rowsB
                B = vertcat(B, NaN(rowsA-rowsB, size(B,2)))
            else
                A = vertcat(A, NaN(rowsB-rowsA, size(A,2)))
            end
        end
        A = horzcat(A,B);
    end
    disp(A)
    
    % write header
    num_tracks = size(trackHistory,2);
    header = {}
   
    for i=1:num_tracks
        header(end+1) = {strcat(strcat('TrackID_',num2str(i)),'_X')};
        header(end+1) = {strcat(strcat('TrackID_',num2str(i)),'_Y')};
    end
    
    % save it in csv file
    data_table = array2table(A, 'VariableNames', header);
    writetable(data_table, strcat(saving, "_RawMeas.csv"));
    
end
    
disp('finished')