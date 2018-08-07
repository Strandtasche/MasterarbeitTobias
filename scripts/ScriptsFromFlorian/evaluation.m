% For more information about the output, look at the save command below.
% All important variables are summarized there.
% @author Florian Pfaff pfaff@kit.edu
% @date 2016-2018
%%
mainDirection='x+'; 

addpath(genpath('.'))
if exist('evalConfig','var')
    predictionEdgePosition=evalConfig.predictionEdgePosition;
    airbarPosition=evalConfig.airbarPosition;
    gtFile=evalConfig.gtFile;
    samplingRateImData=evalConfig.samplingRateImData; %fps
    clearvars evalConfig
else
    predictionEdgePosition=0.638; % Specify position of start of predicion phase
    airbarPosition=0.788; % Specify position of airbar
%     predictionEdgePosition=0.788;airbarPosition=0.788+0.075; % Alterantive config for predicting during flight
    samplingRateImData=1000;
    gtFile='spheres_other_friction_x_y_vx_vy.mat';
%     gtFile='groundtruthPlates.mat';
%     gtFile='groundtruthCylinders.mat';
% % % % %     gtFile='groundtruthSpheres.mat';
% % % %     gtFile='cylinders1.15.mat';
end

% Calibration between image data and ground truth data
A=[ 0.000436681222707424 , 0
    0                    , -0.000437956204379562 ];

t=[0.387563318777293
   0.261897810218978];

beltBordersX=[0.388,0.788];

beltBordersY=[0,0.18];
beltVelocity=1.5*1e-3; % In Meter/Millisekunde
%%
% midpointMatrixTrans=linTransformMidpointMatrix(midpointMatrix,A*[0,1;1,0],t);
% animateMidpointMatrix

addpath(genpath('.'))

if ~exist('gtFileCurr','var')||~strcmp(gtFile,gtFileCurr) % Do not reload if in memory
    if ~exist(gtFile,'file') % Not somewhere incurrent folder, search where
        if exist('D:\TracksortSVN\Daten\DEMSimulation','dir')
            matPath='D:\TracksortSVN\Daten\DEMSimulation';
        else
            disp('Choose folder with groundtruth data');
            matPath=uigetdir;
        end
        addpath(genpath(matPath)); 
    end
    load(gtFile);
    gtFileCurr=gtFile;
end
    
%%
findIntersectionGroundtruth;
%%
assert(size(gt_x_y_vx_vy,3)==size(tracksInfo,2),'Sizes of ground truth and track infos don''t match. When performing predictions based on data read in, also read in ground truth.');
% Reduce data to valid tracks
valid=~any(isnan([tracksInfo.trueIntersectionPosGt]));
gt_x_y_vx_vyVal=gt_x_y_vx_vy(:,:,valid);
tracksInfoVal=tracksInfo(valid);
%%
generatePredictions;
%%

% clear gt_x_y_vx_vy trackHistory % To prevent accidentally using it
if contains(gtFileCurr,'Sphere')
    baseName='Spheres_neu_3x3_dilated_binary_rotated';
elseif contains(gtFileCurr,'Cylinder')
    baseName='zylinder_neu_3x3_dilated_rotated';
elseif contains(gtFileCurr,'Plate')
    baseName='Plates_3x3_dilated_2iter_binary_rotated';
else
    warning('Unknown scenario. Not using image data.');
    baseName='';
end
%%
useImageTracking=false;
if useImageTracking&&~isempty(baseName)
    files=dir(fullfile('Histories',[baseName,num2str(samplingRateImData),'HzHistory*.mat'])); %If multiple versions, take latest
    assert(~isempty(files));
    load(files(end).name)
    %%
    trackHistoryTrans=linTransformTrackHistory(trackHistory,A*[0,1;1,0],t); % Transform with rotation
    findCorrespondences
    saveImgResult=true;
else
    baseName=strrep(gtFileCurr,'.mat','');
    saveImgResult=false;
end

%%
savefn=['Results',baseName,num2str(samplingRateImData),'Hz',num2str(predictionEdgePosition),...
    '-',num2str(airbarPosition),'.mat'];
if saveImgResult
    save(savefn,'imageTrackingResult',... % Results of the tracking based on image data
        'tracksInfoVal',... % Track info including ground truth intersection (only of tracks whose intersection is not empty)
        'tracksInfoValOnbelt',... % Track info of all that are valid and were on belt during the time steps that are recorded in the video (VIDEO DOES NOT COVER ALL TIME STEPS!)
        'trackHistoryTransVal',... % Track History of tracking on video data transformed according to transformation (translation, rotation and scaling to match pixel to world coorindates)
        'tracksPredictions',... % Predictions of all valid tracks
        'tracksPredictionsOnBelt',... % Predictions of those that are on the belt during the time of the video
        'gtFile') % File name read in
else
    save(savefn,'tracksInfoVal',... % Track info including ground truth intersection (only of tracks whose intersection is not empty)
        'tracksPredictions',... % Predictions of all valid tracks
        'gtFile') % File name read in
end