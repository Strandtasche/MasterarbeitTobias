% @author Florian Pfaff pfaff@kit.edu
% @date 2016
downSamplyBy=1000/samplingRateImData;
startAtIndex=7000;
numFrames=4001;
onBelt=squeeze(any(...
    gt_x_y_vx_vyVal(1,startAtIndex:downSamplyBy:startAtIndex+numFrames-1,:)>beltBordersX(1) ...
    & gt_x_y_vx_vyVal(1,startAtIndex:downSamplyBy:startAtIndex+numFrames-1,:)<beltBordersX(2)&...
    gt_x_y_vx_vyVal(2,startAtIndex:downSamplyBy:startAtIndex+numFrames-1,:)>beltBordersY(1) & ...
    gt_x_y_vx_vyVal(2,startAtIndex:downSamplyBy:startAtIndex+numFrames-1,:)<beltBordersY(2)...
    ,2));
tracksInfoValOnbelt=tracksInfoVal(onBelt);
gt_x_y_vx_vyValCutOnBelt=gt_x_y_vx_vyVal(1:4,startAtIndex:downSamplyBy:startAtIndex+numFrames-1,onBelt);
tracksPredictionsOnBelt=tracksPredictions(onBelt);
% clear tracksPredictions tracksInfoValid %prevent usuing it acidentally


%%
% delete invalid tracks
valid=arrayfun(@(t)size(t.Posterior,2),trackHistoryTrans)>=20;
for i=1:length(trackHistoryTrans) 
    if valid(i)
        %Alternative to test for airbar
%         intersection=polyxpoly(airbarPositionX,airbarPositionY,... 
%             trackHistoryTrans(i).Posterior(1,:),trackHistoryTrans(i).Posterior(2,:));
        intersection=polyxpoly(predictionEdgePositionX,predictionEdgePositionY,... 
            trackHistoryTrans(i).Posterior(1,:),trackHistoryTrans(i).Posterior(2,:));
        valid(i)=valid(i)&~isempty(intersection);
    end
%     if valid
%         lastInd=...%cannot work with this track if not <2
%             find(trackHistoryTrans(i).Posterior(1,:)<predictionEdgePosition,1,'last');
%         valid(i)=valid(i)&(lastInd>2);
%     end
end
trackHistoryTransVal=trackHistoryTrans(valid);
% clear trackHistoryTrans
%%

imageTrackingResult=repmat(struct('correspondingIndexGtValidOnbelt',NaN,...
            'lastLocalIndexBeforePrediction',NaN,'lastGlobalTimeBeforePrediction',NaN,...
            'predictedIntersectionTS',NaN(2,1),'predictedTimeTS',NaN,...
            'posDeviationTS',NaN(2,1),'timeErrorTS',NaN,...
            'predictedIntersectionLineImage',NaN(2,1),'posDeviationLineImage',NaN(2,1)),...
        1,numel(trackHistoryTransVal));

trackDists=NaN(length(trackHistoryTransVal),size(gt_x_y_vx_vyValCutOnBelt,3));
for i=1:length(trackHistoryTransVal) 
    if mod(i,50)==0,fprintf('Finding correspondence for track %d of %d\n',i,length(trackHistoryTransVal)),end
    % Version for old matlab
    %diffsForTrack=bsxfun(@minus,trackHistoryTransVal(i).Posterior,...
    %    gt_x_y_vx_vyValCutOnBelt(1:2,trackHistoryTransVal(i).startTime:trackHistoryTransVal(i).LastSeenTime,:));
    % Use implicit expansion for new Matlab
    diffsForTrack=trackHistoryTransVal(i).Posterior-gt_x_y_vx_vyValCutOnBelt(1:2,trackHistoryTransVal(i).startTime:trackHistoryTransVal(i).LastSeenTime,:);
    trackDists(i,:)=sum(sum(abs(diffsForTrack),2),1);
    %%
    [~,imageTrackingResult(i).correspondingIndexGtValidOnbelt]=min(trackDists(i,:));
    plotUntil=3;
    if i==28
        figure(1),clf,hold on
        %axis([min(beltBordersX),max(beltBordersX),min(beltBordersY),max(beltBordersY)])
%         axis([0.544,0.552,0.0578,0.0586])
        axis([0.482,0.49,0.055,0.063])
        plot(gt_x_y_vx_vyValCutOnBelt(1,trackHistoryTransVal(i).startTime:min(trackHistoryTransVal(i).LastSeenTime,plotUntil),imageTrackingResult(i).correspondingIndexGtValidOnbelt)...
            ,gt_x_y_vx_vyValCutOnBelt(2,trackHistoryTransVal(i).startTime:min(trackHistoryTransVal(i).LastSeenTime,plotUntil),imageTrackingResult(i).correspondingIndexGtValidOnbelt),'bx')
        indices=1:min(plotUntil,size(trackHistoryTransVal(i).Tracks_measurment,2));
        plot(trackHistoryTransVal(i).Tracks_measurment(1,indices),trackHistoryTransVal(i).Tracks_measurment(2,indices),'rx')
        drawnow
    end
end
%% Calculate Prediction
for i=1:length(trackHistoryTransVal)
    lastInd=...
        find(trackHistoryTransVal(i).Posterior(1,:)<predictionEdgePosition,1,'last');
    if lastInd<2
        warning('No previous index available. Cannot work with this track.');
        continue
    end
    imageTrackingResult(i).lastLocalIndexBeforePrediction=lastInd;
    imageTrackingResult(i).lastGlobalTimeBeforePrediction=startAtIndex-downSamplyBy+(trackHistoryTransVal(i).startTime+lastInd-1)*downSamplyBy;
    % Calculate prediction deviation TrackSort
    lastVelTS=trackHistoryTransVal(i).Posterior(:,lastInd)...
        -trackHistoryTransVal(i).Posterior(:,lastInd-1);
    farProjektionTS=trackHistoryTransVal(i).Posterior(:,lastInd)+10000*lastVelTS;
    [intXTS,intYTS]=polyxpoly(airbarPositionX,airbarPositionY,... 
            [trackHistoryTransVal(i).Posterior(1,lastInd),farProjektionTS(1)],...
            [trackHistoryTransVal(i).Posterior(2,lastInd),farProjektionTS(2)]);
    imageTrackingResult(i).predictedIntersectionTS=[intXTS;intYTS];
    if isempty(intXTS)
        warning('No intersection found for TS')
        continue
    end
    imageTrackingResult(i).posDeviationTS=imageTrackingResult(i).predictedIntersectionTS-...
        tracksInfoValOnbelt(imageTrackingResult(i).correspondingIndexGtValidOnbelt).trueIntersectionPosGt;
    %%
    % Calculate time (TS)
    distanceToIntTS=norm(imageTrackingResult(i).predictedIntersectionTS-...
        trackHistoryTransVal(i).Posterior(:,lastInd));
    % We have to transform the time to the gt system
    imageTrackingResult(i).predictedTimeTS=imageTrackingResult(i).lastGlobalTimeBeforePrediction+distanceToIntTS/norm(lastVelTS)*downSamplyBy;
    %%
    % Calculate temporal error (TS)
    imageTrackingResult(i).timeErrorTS=...
        tracksInfoValOnbelt(imageTrackingResult(i).correspondingIndexGtValidOnbelt).preciseTimeStepOfIntersection...
        -imageTrackingResult(i).predictedTimeTS;
    %%
    
    
    % Calclate prediction deviation Line (based on last visually observed
    % position)
    farProjektionLineImage=trackHistoryTransVal(i).Posterior(:,lastInd)+10000*beltDirection;
    [intXLine,intYLine]=polyxpoly(airbarPositionX,airbarPositionY,... 
            [trackHistoryTransVal(i).Posterior(1,lastInd),farProjektionLineImage(1)],...
            [trackHistoryTransVal(i).Posterior(2,lastInd),farProjektionLineImage(2)]);
    imageTrackingResult(i).predictedIntersectionLineImage=[intXLine;intYLine];
    if isempty(intXLine)
        warning('No intersection found for line')
        continue
    end
    
    imageTrackingResult(i).posDeviationLineImage=imageTrackingResult(i).predictedIntersectionLineImage-...
        tracksInfoValOnbelt(imageTrackingResult(i).correspondingIndexGtValidOnbelt).trueIntersectionPosGt;
    
    
    
end
%%
assert(abs(mean([imageTrackingResult.timeErrorTS],'omitnan'))<mean([tracksInfoValOnbelt.preciseTimeStepOfIntersection]-[tracksInfoValOnbelt.lastIndexBeforePrediction]),...
    'Temporal deviation is on average longer than the average time between edge and airbar. Something might be wrong..')

%%
if exist('plotting','var')&&plotting
    %%
    posDevTS=sum([imageTrackingResult.posDeviationTS],1);
    posDevLineImage=sum([imageTrackingResult.posDeviationLineImage],1);
    figure(1),clf
    devianBoxPlot1=NaN(2,max(numel(posDevTS),numel(posDevLineImage)));
    devianBoxPlot1(1,1:numel(posDevTS))=posDevTS;
    devianBoxPlot1(2,1:numel(posDevLineImage))=posDevLineImage;
    boxplot(devianBoxPlot1'*1000,{'Predictive Tracking','Prediction Straight Ahead'})
    ylim([-15,15])
    figure(2),clf
    boxplot([posDevTrueVel'*1000,posDevCalVel'*1000,posDevLine'*1000])
    %% plot temporal error
    figure(3),clf
    plot(1:numel(imageTrackingResult),[imageTrackingResult.timeErrorTS])
    figure(4),clf
    boxplot(-[imageTrackingResult.timeErrorTS])

end