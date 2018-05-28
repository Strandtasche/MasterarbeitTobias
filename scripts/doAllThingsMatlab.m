disp('starting script')
allParam=getDefaultParam([30;0]);
allParam.score.looseStep=30;
allParam.association.useOrientation=false;
allParam.initial.PositionCov=2000*eye(2);
allParam.meas.PositionCov=2000*eye(2);
allParam.association.tryToUseMex=false;
disp('parameters set!')

files = dir('*_data.csv');
i = 1;
for file = files'
    trackHistory=tracksortAlgorithm([0,2450;0,1500],1300,1450,allParam,file.name);
    saving = strcat("Trackhistory_", file.name(1:end-4));
    saving = strcat(saving, ".mat")
    save(saving, "trackHistory")
    i = i + 1;
    if i == 4
        a = 42;
    end
end
    
disp('finished')