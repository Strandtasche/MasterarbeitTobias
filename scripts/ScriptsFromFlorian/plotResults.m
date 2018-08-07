% return % do not let this run because all plots will be overwritten
% if ~exist('filename','var')||isequal('filename',0) %#ok<UNRCH>
%     filename=uigetfile;
% end
% load(filename);
% return
%% Spatial and temporal for ground truth and one data set
% This requires that alignment may not be at an arbitrary angle and that the deviation may
% either be on x OR on y direction - in this case, one of the numbers is 0,
% the other is the deviation.
posDevTrueVel=sum([tracksPredictions.posDeviationTrueVel],1);
posDevCalVel=sum([tracksPredictions.posDeviationCalVel],1);
posDevLine=sum([tracksPredictions.posDeviationLine],1);
posDevConstAcc=sum([tracksPredictions.posDeviationConstAcc],1);
posDevMeanOverAll=sum([tracksPredictions.posDeviationMeanOverAll],1);
posDevLimAcc=sum([tracksPredictions.posDeviationLimAcc],1);
posDevLimAccNSC=sum([tracksPredictions.posDeviationLimAccNSC],1);
posDevConstVelCorr=sum([tracksPredictions.posDeviationConstVelCorr],1);
posDevConstVelCorrXConstAccY=sum([tracksPredictions.posDeviationConstVelCorrXConstAccY],1);
posDevCVIA_CV=sum([tracksPredictions.posDeviationCVIA_CV],1);
posDevCVIA_CA=sum([tracksPredictions.posDeviationCVIA_CA],1);
posDevCVIA_Scale=sum([tracksPredictions.posDeviationCVIA_Scale],1);
posDevCVIA_Ratio=sum([tracksPredictions.posDeviationCVIA_Ratio],1);

if exist('imageTrackingResult','var')
    posDevTS=sum([imageTrackingResult.posDeviationTS],1);
    posDevLineImage=sum([imageTrackingResult.posDeviationLineImage],1);
end

timeDevTrueVel=[tracksPredictions.timeErrorTrueVel];
timeDevCalVel=[tracksPredictions.timeErrorCalVel];
timeDevConstAcc=[tracksPredictions.timeErrorConstAcc];
timeDevMeanOverAll=[tracksPredictions.timeErrorMeanOverAll];
timeDevMedianOverAll=[tracksPredictions.timeErrorMedianOverAll];
timeDevLimAcc=[tracksPredictions.timeErrorLimAcc];
timeDevConstVelCorr=[tracksPredictions.timeErrorConstVelCorr];
timeDevCVIA=[tracksPredictions.timeErrorCVIA];
timeDevRatio=[tracksPredictions.timeErrorRatio];

if isempty(mfilename) % plot only if executed in cell mode
    % Spatial error sim
    figure(1),shg,clf
    % boxplot(1000*[posDevTrueVelSph',posDevCalVelSph',posDevLineSph'],{'Sim','Tracking','Straight'},'symbol','')% MFI 2016
    % boxplot(1000*[posDevTrueVelSph',posDevCalVelSph',posDevLineSph',posDevConstAccSph'],{'Sim','Tracking','Straight','ConstAcc'},'symbol','')
    boxplot(1000*[posDevLine',posDevTrueVel',posDevCalVel',posDevConstAcc',posDevMeanOverAll',posDevLimAcc',posDevConstVelCorr',posDevConstVelCorrXConstAccY'],...
        {'Straight','ConVel(sim)','ConVel(est)','ConAcc','Mean','LimAcc','ConVelCorr','CombinedTest'},'symbol','')
    % ylim([-1.25,1.25])
    ylabel('Spatial deviation in mm')
    % temporal error sim
    figure(11),shg,clf
    % boxplot([timeDevTrueVelSph',timeDevCalVelSph'],{'Sim','Tracking'},'symbol','')
    % boxplot([timeDevTrueVelSph',timeDevCalVelSph',timeDevConstAccSph'],{'Sim','Tracking','Const Acc'},'symbol','')
    % boxplot([timeDevTrueVelSph',timeDevCalVelSph',timeDevConstAccSph',timeDevMeanOverAllSph'],{'Sim','Tracking','Const Acc','MeanOverAll'},'symbol','')
    %for mean vel corr: timeDevCalVel'-mean(timeDevCalVel(1:floor(numel(timeDevCalVel)*0.01)))
    boxplot(-[timeDevMeanOverAll',timeDevMedianOverAll',timeDevTrueVel',timeDevCalVel',timeDevConstAcc',timeDevLimAcc',timeDevConstVelCorr'],...
        {'IdDel(mean)','IdDel(median)','ConVel(sim)','ConVel(est)','ConAcc','LimAcc','ConstVelCorr'},'symbol','')
    ylabel('Temporal deviation in ms')
    % ylim([-3,3])
end
return

%% Spatial and temporal for image data and one data set
figure(2),shg,clf
boxplotFromCell({1000*posDevTS,1000*posDevLineImage});
ylim([-1.5,2])
ylabel('Spatial deviation in mm')


%% Temporal deviation with image processing
timeDevTSSph=[imageTrackingResult.timeErrorTS];
figure(3)
boxplot(-timeDevTSSph',{'Tracking'},'symbol','')
ylim([0,2.5])
ylabel('Temporal deviation in ms')



%% Saving all plots
figure(1)
prepareFig([4,5])
export_fig ../resultsSimTrackingLineGt.pdf

figure(2)
prepareFig([3.5,5])
export_fig ../resultsTrackingLine.pdf

figure(3)
prepareFig([3,5])
export_fig ../resultsTimeIm.pdf


%% Compare spheres with cylinders 6 cm prediction phase
figure(1),clf,hold on
HzToPlot=1000;
load(sprintf('ResultsSpheres_neu_3x3_dilated_binary_rotated%dHz0.688-0.788.mat',HzToPlot));
posDevTrueVelSph=sum([tracksPredictions.posDeviationTrueVel],1);
posDevCalVelSph=sum([tracksPredictions.posDeviationCalVel],1);
load(sprintf('Resultszylinder_neu_3x3_dilated_rotated%dHz0.688-0.788.mat',HzToPlot));
posDevTrueVelCyl=sum([tracksPredictions.posDeviationTrueVel],1);
posDevCalVelCyl=sum([tracksPredictions.posDeviationCalVel],1);

boxplotFromCell({1000*posDevTrueVelSph,1000*posDevCalVelSph,1000*posDevTrueVelCyl,1000*posDevCalVelCyl},{'Spheres: Sim','Spheres: Tracking',...
    'Cylinders: Sim','Cylinders: Tracking'})
ylim([-1,1])
ylabel('Deviation along y-axis in mm')

%% TII Plots
%% Spheres spatial
load ResultsSpheres_neu_3x3_dilated_binary_rotated1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(1),shg,clf
boxplot(1000*[posDevLine',posDevCalVel',posDevConstAcc'],{'Straight','Const. Vel.','Const. Acc.'},'symbol','')
ylim([-1.9,1.9])
ylabel('Spatial deviation in mm')
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\resultsSpheresSpatial.pdf
%% cylinders spatial
load Resultszylinder_neu_3x3_dilated_rotated1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(1),shg,clf
boxplot(1000*[posDevLine',posDevCalVel',posDevConstAcc'],{'Straight','Const. Vel.','Const. Acc.'},'symbol','')
ylim([-4.8,4.8])
ylabel('Spatial deviation in mm')
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\resultsCylindersSpatial.pdf
%% plates spatial
load ResultsPlates_3x3_dilated_2iter_binary_rotated1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(1),shg,clf
% boxplot(1000*[posDevCalVel',posDevLine',posDevConstAcc'],{'Const. Vel.','Straight','Const. Acc.'},'symbol','')
boxplot(1000*[posDevLine',posDevCalVel',posDevConstAcc'],{'Straight','Const. Vel.','Const. Acc.'},'symbol','')
ylim([-1.5,1.5])
ylabel('Spatial deviation in mm')
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\resultsPlatesSpatial.pdf
%% spheres temporal
load ResultsSpheres_neu_3x3_dilated_binary_rotated1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
boxplot(-[timeDevMeanOverAll',timeDevCalVel',timeDevConstAcc',timeDevCalVel'-mean(timeDevCalVel(1:floor(numel(timeDevCalVel)*0.01)))],...
    {'MeanOverAll','Const. Vel.','Const. Acc.','Const. vel. IA'},'symbol','')
ylabel('Temporal deviation in ms')
ylim([-3.6,4.7]);
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\resultsSpheresTemporal.pdf

%% cylidners temporal
load Resultszylinder_neu_3x3_dilated_rotated1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
boxplot(-[timeDevMeanOverAll',timeDevCalVel',timeDevConstAcc',timeDevCalVel'-mean(timeDevCalVel(1:floor(numel(timeDevCalVel)*0.01)))],...
    {'MeanOverAll','Const. Vel.','Const. Acc.','Const. vel. IA'},'symbol','')
ylabel('Temporal deviation in ms')
ylim([-4,9])
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\resultsCylindersTemporal.pdf

%% plates temporal
load ResultsPlates_3x3_dilated_2iter_binary_rotated1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
boxplot(-[timeDevMeanOverAll',timeDevCalVel',timeDevConstAcc',timeDevCalVel'-mean(timeDevCalVel(1:floor(numel(timeDevCalVel)*0.01)))],...
    {'MeanOverAll','Const. Vel.','Const. Acc.','Const. vel. IA'},'symbol','')
ylabel('Temporal deviation in ms')
ylim([-4,9])
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\resultsPlatesTemporal.pdf



%%% NEW for TII
%%
plotSize=[10.25,5];
%% spheres temporal other friction
load Resultsspheres_other_friction_x_y_vx_vy1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(1);
boxplot(-[timeDevMedianOverAll',timeDevCalVel',timeDevConstVelCorr',timeDevLimAccNSC'],...
    {'Id. delay','Const. vel.','CVIA','CALV'},'symbol','')
ylabel('Temporal deviation in ms')
ylim([-1.5,3.5]);
% title('Spheres (new friction)');
prepareFig(plotSize)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\resultsSpheresOtherFricTemporal.pdf
%% spheres spatial
load Resultsspheres_other_friction_x_y_vx_vy1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(2);
boxplot(1000*[posDevLine',posDevCalVel',posDevConstVelCorr',posDevLimAccNSC',posDevConstVelCorrXConstAccY'],...
    {'Straight','Const. vel.','CVIA','CANC','Combination'},'symbol','')
ylabel('Spatial deviation in mm')
ylim([-1.7,1.7])
% title('Spheres (new friction)');
prepareFig(plotSize)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\resultsSpheresOtherFricSpatial.pdf
%% plates temporal
load ResultsgroundtruthPlates1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(11);
boxplot(-[timeDevMedianOverAll',timeDevCalVel',timeDevConstVelCorr',timeDevLimAccNSC'],...
    {'Id. delay','Const. vel.','CVIA','CALV'},'symbol','')
ylabel('Temporal deviation in ms')
ylim([-4,8.5]);
% title('Plates');
prepareFig(plotSize)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\resultsCuboidsTemporal.pdf
%% plates spatial
load ResultsgroundtruthPlates1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(12);
boxplot(1000*[posDevLine',posDevCalVel',posDevConstVelCorr',posDevLimAccNSC',posDevConstVelCorrXConstAccY'],...
    {'Straight','Const. vel.','CVIA','CANC','Combination'},'symbol','')
ylabel('Spatial deviation in mm')
ylim([-1.6,1.6])
% title('Plates');
prepareFig(plotSize)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\resultsCuboidsSpatial.pdf
%% cylinders temporal
load ResultsgroundtruthCylinders1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(21);
boxplot(-[timeDevMedianOverAll',timeDevCalVel',timeDevConstVelCorr',timeDevLimAccNSC'],...
    {'Id. delay','Const. vel.','CVIA','CALV'},'symbol','')
ylabel('Temporal deviation in ms')
ylim([-4,8.5]);
% title('Cylinders');
prepareFig(plotSize)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\resultsCylindersTemporal.pdf
%% cylinders spatial
load ResultsgroundtruthCylinders1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(22);
boxplot(1000*[posDevLine',posDevCalVel',posDevConstVelCorr',posDevLimAccNSC',posDevConstVelCorrXConstAccY'],...
    {'Straight','Const. vel.','CVIA','CANC','Combination'},'symbol','')
ylabel('Spatial deviation in mm')
ylim([-4.8,4.8])
prepareFig(plotSize)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\resultsCylindersSpatial.pdf

%%% NEW for AES
%%
plotSize=[13,4.1];
plotSizeSpatial=[16,4.1];
% spheres temporal other friction
load Resultsspheres_other_friction_x_y_vx_vy1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(1);
boxplot(-[timeDevMedianOverAll',timeDevCalVel',timeDevConstVelCorr',timeDevConstAcc',timeDevLimAcc',timeDevCVIA'],...
    {'IV','CV','CVBC','CA','CALV','IA'},'symbol','')
ylabel('Temporal deviation in ms')
ylim([-1.5,3.5]);
% title('Spheres (new friction)');
prepareFig(plotSize)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\Draft2AEI\resultsSpheresOtherFricTemporal.pdf
% plates temporal
load ResultsgroundtruthPlates1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(11);
boxplot(-[timeDevMedianOverAll',timeDevCalVel',timeDevConstVelCorr',timeDevConstAcc',timeDevLimAcc',timeDevCVIA'],...
    {'IV','CV','CVBC','CA','CALV','IA'},'symbol','')
ylabel('Temporal deviation in ms')
ylim([-4,8.5]);
% title('Plates');
prepareFig(plotSize)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\Draft2AEI\resultsCuboidsTemporal.pdf
% cylinders temporal
load ResultsgroundtruthCylinders1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(21);
boxplot(-[timeDevMedianOverAll',timeDevCalVel',timeDevConstVelCorr',timeDevConstAcc',timeDevLimAcc',timeDevCVIA'],...
    {'IV','CV','CVBC','CA','CALV','IA'},'symbol','')
ylabel('Temporal deviation in ms')
ylim([-4,8.5]);
% title('Cylinders');
prepareFig(plotSize)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\Draft2AEI\resultsCylindersTemporal.pdf
% other plot size for spatial
% plotSize=[16,5];
% plotSizeSpatial=[16,3];

% spheres spatial
load Resultsspheres_other_friction_x_y_vx_vy1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(2);
boxplot(1000*[posDevLine',posDevCalVel',posDevLimAcc',posDevLimAccNSC',posDevCVIA_CV',posDevCVIA_CA',posDevCVIA_Ratio',posDevCVIA_Scale'],...
    {'Straight','CV-CV','CALV-CA','CALV-CANSC','IA-CV','IA-CA','IA-ratio','IA-scale'},'symbol','')
ylabel('Spatial deviation in mm')
ylim([-1.7,1.7])
% title('Spheres (new friction)');
prepareFig(plotSizeSpatial)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\Draft2AEI\resultsSpheresOtherFricSpatial.pdf

% plates spatial
load ResultsgroundtruthPlates1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(12);
boxplot(1000*[posDevLine',posDevCalVel',posDevLimAcc',posDevLimAccNSC',posDevCVIA_CV',posDevCVIA_CA',posDevCVIA_Ratio',posDevCVIA_Scale'],...
    {'Straight','CV-CV','CALV-CA','CALV-CANSC','IA-CV','IA-CA','IA-ratio','IA-scale'},'symbol','')
ylabel('Spatial deviation in mm')
ylim([-1.6,1.6])
% title('Plates');
prepareFig(plotSizeSpatial)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\Draft2AEI\resultsCuboidsSpatial.pdf

% cylinders spatial
load ResultsgroundtruthCylinders1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(22);
boxplot(1000*[posDevLine',posDevCalVel',posDevLimAcc',posDevLimAccNSC',posDevCVIA_CV',posDevCVIA_CA',posDevCVIA_Ratio',posDevCVIA_Scale'],...
    {'Straight','CV-CV','CALV-CA','CALV-CANSC','IA-CV','IA-CA','IA-ratio','IA-scale'},'symbol','')
ylabel('Spatial deviation in mm')
ylim([-4.8,4.8])
prepareFig(plotSizeSpatial)
export_fig -transparent ..\..\TBD17_Pfaff-TrackSortImprovedModels\Draft2AEI\resultsCylindersSpatial.pdf

%% Präsi PA temporal
plotSize=[13,4.1];
% spheres temporal other friction
% load Resultsspheres_other_friction_x_y_vx_vy1000Hz0.638-0.788.mat
load ResultsgroundtruthCylinders1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(1);
boxplot(-[timeDevMedianOverAll',timeDevCalVel',timeDevConstAcc',timeDevCVIA'],...
    {'Const. vel.','Const. acc.','Id. vel.','Id. acc'},'symbol','')
% boxplot(-[timeDevCalVel',timeDevConstAcc',timeDevMedianOverAll',timeDevCVIA'],...
%     {'a','b','c','D'},'symbol','')
% ylabel('Zeitliche Abweichung in ms')
% ylim([-1.5,3.5]);
ylim([-4,8.5]);
% title('Spheres (new friction)');
prepareFig(plotSize,2,9,0.6,false,false)
allText=findall(gca, 'Type', 'Text');
% set(allText,'FontSize', fontSize*scaling);
fontName='Helvetica';
set(findall(gcf, 'Type', 'Axes'), 'FontName', fontName);
set(allText(isvalid(allText)), 'FontName', fontName);
set(allText,'FontName', 'Helvetica');
export_fig -transparent ../../resultsSpheresOtherFricTemporalPA.png

%% Präsi PA spheres spatial
plotSize=[8.5,4.1];
% spheres temporal other friction
% load Resultsspheres_other_friction_x_y_vx_vy1000Hz0.638-0.788.mat
load Resultsspheres_other_friction_x_y_vx_vy1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(2);
boxplot(1000*[posDevLine',posDevCalVel',posDevConstAcc',posDevCVIA_Ratio',posDevCVIA_Scale'],...
    {'Straight','CV','CA-CA','IA-ratio','IA-scale'},'symbol','')
% ylabel('Spatial deviation in mm')
ylim([-1.7,1.7])
prepareFig(plotSize,2,9,0.6,false,false)
fontName='Helvetica';
set(findall(gcf, 'Type', 'Axes'), 'FontName', fontName);
set(allText(isvalid(allText)), 'FontName', fontName);
set(allText,'FontName', 'Helvetica');
export_fig -transparent ../../resultsSpheresOtherFricSpatialPA.png
plotResults % Reclculate the values for the boxplots
%% Präsi PA cylinders spatial
plotSize=[8.5,4.1];
% spheres temporal other friction
load ResultsgroundtruthCylinders1000Hz0.638-0.788.mat
plotResults % Reclculate the values for the boxplots
figure(2);
boxplot(1000*[posDevLine',posDevCalVel',posDevConstAcc',posDevCVIA_Ratio',posDevCVIA_Scale'],...
    {'Straight','CV','CA-CA','IA-ratio','IA-scale'},'symbol','')
% ylabel('Spatial deviation in mm')
ylim([-4.8,4.8])
prepareFig(plotSize,2,9,0.6,false,false)
fontName='Helvetica';
set(findall(gcf, 'Type', 'Axes'), 'FontName', fontName);
set(allText(isvalid(allText)), 'FontName', fontName);
set(allText,'FontName', 'Helvetica');
export_fig -transparent ../../resultsCylindersSpatialPA.png
plotResults % Reclculate the values for the boxplots
%% Comparing all models
% !!!!!!! load mat before executing
plotResults % Reclculate the values for the boxplots
figure(1);clf
boxplot(-[timeDevMedianOverAll',timeDevTrueVel',timeDevCalVel',timeDevConstVelCorr',timeDevConstAcc',timeDevLimAcc',timeDevCVIA',timeDevRatio'],...
    {'Id. delay','CV Ideal','CV','CV Corr','CA','CALV','CVIA','CVRatio'})%,'symbol','')
ylabel('Temporal deviation in ms')
% ylim([-10,10]);
% prepareFig(plotSize)
figure(2);clf
boxplot(1000*[posDevLine',posDevTrueVel',posDevCalVel',posDevConstVelCorr',posDevConstVelCorrXConstAccY',posDevConstAcc',posDevLimAcc',posDevLimAccNSC',posDevCVIA_CV',posDevCVIA_CA',posDevCVIA_Scale',posDevCVIA_Ratio'],...
    {'Straight','CV Ideal','CV-CV','CVBC-CV','CVBC-CA','CA-CA','CALV-CA','CALV-CANSC','CVIA-CV','CVIA-CA','CVIA-Scale','CVIA-Ratio'})%,'symbol','')
ylabel('Spatial deviation in mm')
ylim([-5,5])
% prepareFig(plotSize)
% figure(10)
% hist(posDevConstVelCorr',1000)
% hist(posDevCVIA_CV',2000)
% hist(posDevCVIA_Scale',2000)

