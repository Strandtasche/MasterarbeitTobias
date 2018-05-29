% Usage: 
% 1. add nessesary things to path -addpath(genpath('matlab') ?
% 2. navigate to folder with data you want to process
% 3. optional: change files variable to fit your input csvs.
% >> doAllThingsMatlab

disp('starting script')

%Set magic Parameters, so the tracking works well
allParam=getDefaultParam([30;0]);
allParam.score.looseStep=30;
allParam.association.useOrientation=false;
allParam.initial.PositionCov=2000*eye(2);
allParam.meas.PositionCov=2000*eye(2);
allParam.association.tryToUseMex=false;
disp('parameters set!')

%files = dir('*_data.csv');
files = dir('*case3.csv');
i = 1;
for file = files'
    trackHistory=tracksortAlgorithm([0,2450;0,1500],1300,1450,allParam,file.name);
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
        A = horzcat(A,B);
    end
    disp(A)
    %csvwrite(strcat(saving,"_RawMeas.csv") ,trackHistory(1).RawMeasurements)
    
end
    
disp('finished')