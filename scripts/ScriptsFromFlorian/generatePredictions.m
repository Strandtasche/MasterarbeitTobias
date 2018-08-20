% @author Florian Pfaff pfaff@kit.edu
% @date 2016-2018
% Generates predictions
tracksPredictions=repmat(struct(...
    'predictedIntersectionTrueVel',NaN(2,1),'posDeviationTrueVel',NaN(2,1),...
    'predictedTimeTrueVel',NaN,'timeErrorTrueVel',NaN,... % Time error: Minus means later, plus means earlier
    'predictedIntersectionCalVel',NaN(2,1),'posDeviationCalVel',NaN(2,1),...
    'predictedTimeCalVel',NaN,'timeErrorCalVel',NaN,...
    'predictedIntersectionLine',NaN(2,1),'posDeviationLine',NaN(2,1),... % Indepedent of acc
    'predictedIntersectionConstAcc',NaN(2,1),'posDeviationConstAcc',NaN(2,1),...
    'predictedTimeConstAcc',NaN,'timeErrorConstAcc',NaN,...
    ...% Intersection calculated for median and mean over all velocities as well (only useful if any bias along y)
    'predictedIntersectionMeanOverAll',NaN(2,1),'posDeviationMeanOverAll',NaN(2,1),... 
    'predictedTimeMeanOverAll',NaN,'timeErrorMeanOverAll',NaN,...
    'predictedIntersectionMedianOverAll',NaN(2,1),'posDeviationMedianOverAll',NaN(2,1),...
    'predictedTimeMedianOverAll',NaN,'timeErrorMedianOverAll',NaN,...
    'predictedIntersectionLimAcc',NaN(2,1),'posDeviationLimAcc',NaN(2,1),...
    'predictedTimeIdenticalDelayMedian',NaN,'timeErrorIdenticalDelayMedian',NaN,...
    'predictedTimeLimAcc',NaN,'timeErrorLimAcc',NaN,...
    ...% Time for LimAccNSC is identical to LimAcc
    'predictedIntersectionLimAccNSC',NaN(2,1),'posDeviationLimAccNSC',NaN(2,1),...
    'predictedIntersectionConstVelCorr',NaN(2,1),'posDeviationConstVelCorr',NaN(2,1),...
    'predictedTimeConstVelCorr',NaN,'timeErrorConstVelCorr',NaN,...
    ...% Time for ConstVelCorrXConstAccY is identical to ConstVel
    'predictedIntersectionConstVelCorrXConstAccY',NaN(2,1),'posDeviationConstVelCorrXConstAccY',NaN(2,1),...
    'predictedTimeCVIA',NaN,'timeErrorCVIA',NaN,...
    ...% Combining CVIA with different y-axis models
    'predictedIntersectionCVIA_CV',NaN(2,1),'posDeviationCVIA_CV',NaN(2,1),...
    'predictedIntersectionCVIA_CA',NaN(2,1),'posDeviationCVIA_CA',NaN(2,1),...
    'predictedIntersectionCVIA_Scale',NaN(2,1),'posDeviationCVIA_Scale',NaN(2,1),...
    'predictedIntersectionCVIA_Ratio',NaN(2,1),'posDeviationCVIA_Ratio',NaN(2,1),...
    'predictedTimeRatio',NaN,'timeErrorRatio',NaN),...
    [1,size(gt_x_y_vx_vyVal,3)]);
%% Calculate some statistics for the mean over all model.
firstIndicesAfterPred=[tracksInfoVal.lastIndexBeforePrediction]+1;
firstIndicesAfterInt=ceil([tracksInfoVal.preciseTimeStepOfIntersection]);
firstPos=zeros(2,numel(firstIndicesAfterPred));
lastPos=zeros(2,numel(firstIndicesAfterPred));
for i=1:numel(firstIndicesAfterPred) % Average velocity over prediction region
    assert(~isnan(tracksInfoVal(i).trueIntersectionPosGt(1)),'No intersection for GT! Only use valid tracks here')
    firstPos(:,i)=gt_x_y_vx_vyVal(1:2,firstIndicesAfterPred(i),i);
    lastPos(:,i)=gt_x_y_vx_vyVal(1:2,firstIndicesAfterInt(i),i);
    velPerStep=(lastPos-firstPos)./(firstIndicesAfterInt-firstIndicesAfterPred); % Use implicit expansion
end
meanVelPerStep=mean(velPerStep,2);
medianVelPerStep=median(velPerStep,2);
if abs(meanVelPerStep(2))>1e-4
    warning('Particles are not flying straight on average. This is unexpected behavior');
end
%%
velocitiesCalculated=NaN(2,size(gt_x_y_vx_vyVal,3));
% Avoid bugs due to values in previous iterations
clear *BeforePred distAlongBelt trueVelBeforePred farProjection*  distanceToInt* timeToSep beforelastVelBeforePred changeAccBeforePred predictedPositions int* allPossibleTimes
for i=1:size(gt_x_y_vx_vyVal,3) 
    if mod(i,50)==0,fprintf('Predicting based on GT track %d of %d\n',i,size(gt_x_y_vx_vyVal,3)),end
    assert(~isnan(tracksInfoVal(i).trueIntersectionPosGt(1)),'No intersection for GT! Only use valid tracks here')
    
    lastPosBeforePred=gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction,i);
    distAlongBelt=airbarPositionX(1)-lastPosBeforePred(1);
    
    % First calculate everything based on the true veloctiy known at this
    % instant (TrueVel)
    % Calculate pos (TrueVel)
    trueVelBeforePred=gt_x_y_vx_vyVal(3:4,tracksInfoVal(i).lastIndexBeforePrediction,i)/1000;
    
    timeToSep=roots([trueVelBeforePred(1),-distAlongBelt]);
    if numel(timeToSep)~=1
        error('Could not calculate time to separation for CV model. This should not happen.');
    else
        % If found, save time and calculate predicted position.
        tracksPredictions(i).predictedTimeTrueVel=tracksInfoVal(i).lastIndexBeforePrediction+timeToSep;
        tracksPredictions(i).predictedIntersectionTrueVel=...
            [airbarPositionX(1);lastPosBeforePred(2)+trueVelBeforePred(2)*timeToSep]; % X coorindate is obvious
    end
    % Calculate pos error (TrueVel)
    tracksPredictions(i).posDeviationTrueVel=...
        tracksPredictions(i).predictedIntersectionTrueVel - tracksInfoVal(i).trueIntersectionPosGt;
    % Calculate time (TrueVel)
    % Calculate temporal error (TrueVel)
    tracksPredictions(i).timeErrorTrueVel=...
        tracksInfoVal(i).preciseTimeStepOfIntersection-tracksPredictions(i).predictedTimeTrueVel;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Now calculate everything based on the last two positions CalVel) (since meas
    % uncertainty is zero, this is what our filter would result in)
    % Calculate pos (CalVel)
    calVelBeforePred=lastPosBeforePred-gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-1,i);
    velocitiesCalculated(:,i)=calVelBeforePred; % Save for const vel corrected

    timeToSepCV=roots([calVelBeforePred(1),-distAlongBelt]);
    if numel(timeToSepCV)~=1
        warning('Did not find intersection for constant acceleration. Setting to NaN');
    else
        % If found, save time and calculate predicted position.
        tracksPredictions(i).predictedTimeCalVel=tracksInfoVal(i).lastIndexBeforePrediction+timeToSepCV;
        tracksPredictions(i).predictedIntersectionCalVel=...
            [airbarPositionX(1);lastPosBeforePred(2)+calVelBeforePred(2)*timeToSepCV]; % X coorindate is obvious
    end
   
    % Calculate pos error (CalVel)
    tracksPredictions(i).posDeviationCalVel=...
        tracksPredictions(i).predictedIntersectionCalVel - tracksInfoVal(i).trueIntersectionPosGt;
    % Calculate time (CalVel)
    tracksPredictions(i).timeErrorCalVel=...
        tracksInfoVal(i).preciseTimeStepOfIntersection-tracksPredictions(i).predictedTimeCalVel;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Now perform calculation for constant acceleration model. We calculate
    % the velocities and their difference as the accerlation (this would be
    % the filter result as there is no measurement uncertainty). Due to the
    % acceleration component, the trajectory is generally not a line (just
    % imagine there being only an acceleration orthogonal to the transport
    % direction). Therefore, the intersection test has to be adapted to
    % respect the actual trajectory. Furthermore, multiple intersections
    % are possible for the constant acceleration model as a deceleration
    % can result in a reversal of the movement.
    lastVelBeforePred=lastPosBeforePred-gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-1,i);
    beforelastVelBeforePred=gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-1,i)-gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-2,i);
    lastAccBeforePred=lastVelBeforePred-beforelastVelBeforePred;
    
    allPossibleTimes=roots([0.5*lastAccBeforePred(1),lastVelBeforePred(1),-distAlongBelt]);
    timeToSepCA=min(allPossibleTimes(allPossibleTimes>=0));
    if imag(timeToSepCA)>0.00001
        error('Obtained imginary value where real value was expected');
    else
        timeToSepCA=real(timeToSepCA);
    end
    
    if isempty(timeToSepCA)
        warning('Did not find intersection for constant acceleration. Setting to NaN');
    else
        % If found, save time and calculate predicted position.
        tracksPredictions(i).predictedTimeConstAcc=tracksInfoVal(i).lastIndexBeforePrediction+timeToSepCA;
        tracksPredictions(i).predictedIntersectionConstAcc=...
            [airbarPositionX(1);...
            lastPosBeforePred(2)+calVelBeforePred(2)*timeToSepCA+0.5*lastAccBeforePred(2)*timeToSepCA^2]; % X coorindate is obvious
    end
    
    % Calculate temporal error (Const Acc)
    tracksPredictions(i).timeErrorConstAcc=...
        tracksInfoVal(i).preciseTimeStepOfIntersection-tracksPredictions(i).predictedTimeConstAcc;
    tracksPredictions(i).posDeviationConstAcc=...
        tracksPredictions(i).predictedIntersectionConstAcc - tracksInfoVal(i).trueIntersectionPosGt;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Now perform calculation for limited acceleration model. We assume
    % that the velocity along the axis orthogonal to the transport
    % direction never changes sign. We regard two models for the y-axis.
    % One in which we disallow a change in sign and one at which we allow
    % it. Belt velocity is in Meter pro Millisekunde, our velocity is in
    % m/time step (der auch 1 ms ist bei Standardkonfig)
    surpassedBeltSpeedAt=(beltVelocity-lastVelBeforePred(1))/lastAccBeforePred(1);
    if timeToSepCA<surpassedBeltSpeedAt 
        timeToSepCALV=timeToSepCA; % Can use regular CA
    else
        % Reduce distance by distance traveled until beltVelocity is
        % reached. Remaining distance is traveled by the speed of
        % beltVeocity
        timeToSepCALV=surpassedBeltSpeedAt+(distAlongBelt-lastVelBeforePred(1)*surpassedBeltSpeedAt-lastAccBeforePred(1)*surpassedBeltSpeedAt^2)/beltVelocity;
    end
    tracksPredictions(i).predictedTimeLimAcc=tracksInfoVal(i).lastIndexBeforePrediction+timeToSepCALV;
    
    % First when allowing change in sign
    tracksPredictions(i).predictedIntersectionLimAcc=...
            [airbarPositionX(1);...
            lastPosBeforePred(2)+calVelBeforePred(2)*timeToSepCALV+0.5*lastAccBeforePred(2)*timeToSepCALV^2]; % X coorindate is obvious
        
    % Determine when sign changes
    signChangeAt=lastVelBeforePred(2)/lastAccBeforePred(2);
    if (signChangeAt<0)||(signChangeAt<=timeToSepCALV)||isinf(signChangeAt)||isnan(signChangeAt)
        % Can simply use formulae for CA (but with time of CALV)
        tracksPredictions(i).predictedIntersectionLimAccNSC=tracksPredictions(i).predictedIntersectionLimAcc;
    else
        % Position along y-axis only changes until sign change. Can use
        % formula but use signChangeAt instead of timeToSepCALV
        tracksPredictions(i).predictedIntersectionLimAccNSC=...
            [airbarPositionX(1);...
            lastPosBeforePred(2)+calVelBeforePred(2)*signChangeAt+0.5*lastAccBeforePred(2)*signChangeAt^2]; % X coorindate is obvious
    end
    assert(~any(isnan(tracksPredictions(i).predictedIntersectionLimAccNSC)));
    tracksPredictions(i).posDeviationLimAcc=...
        tracksPredictions(i).predictedIntersectionLimAcc - tracksInfoVal(i).trueIntersectionPosGt;
    tracksPredictions(i).posDeviationLimAccNSC=...
        tracksPredictions(i).predictedIntersectionLimAccNSC - tracksInfoVal(i).trueIntersectionPosGt;
    tracksPredictions(i).timeErrorLimAcc=...
        tracksInfoVal(i).preciseTimeStepOfIntersection-tracksPredictions(i).predictedTimeLimAcc;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Now perform calculation for a statistically based model that assumes
    % that all particles take the same time and arrive at the same offset
    % (the spatial offset should be zero for a perfect belt).
    % We assume that we observe (at least some) particles over the whole
    % region before this model can be used. Therefore, we assume that we
    % can use some statistics that also take steps after the prediction
    % into account in the statistics. We use the first time step after the
    % beginning of the prediction and the last time step after the
    % separation as baseline (this choice could be changed).
    farProjectionMeanOverAll=lastPosBeforePred+10000*meanVelPerStep;
    
    [tracksPredictions(i).predictedIntersectionMeanOverAll(1),tracksPredictions(i).predictedIntersectionMeanOverAll(2)]...
        =polyxpoly(airbarPositionX,airbarPositionY,...
        [lastPosBeforePred(1),farProjectionMeanOverAll(1)],[lastPosBeforePred(2),farProjectionMeanOverAll(2)]);
    % Calculate pos error (MeanOverAll)
    tracksPredictions(i).posDeviationMeanOverAll=...
        tracksPredictions(i).predictedIntersectionMeanOverAll - tracksInfoVal(i).trueIntersectionPosGt;
    % Calculate time (MeanOverAll)
    distanceToIntMeanOverAll=norm(lastPosBeforePred-tracksPredictions(i).predictedIntersectionMeanOverAll);
    tracksPredictions(i).predictedTimeMeanOverAll=tracksInfoVal(i).lastIndexBeforePrediction+...
        distanceToIntMeanOverAll/(norm(meanVelPerStep)); % Is already in m / ms
    % Calculate temporal error (MeanOverAll)
    tracksPredictions(i).timeErrorMeanOverAll=...
        tracksInfoVal(i).preciseTimeStepOfIntersection-tracksPredictions(i).predictedTimeMeanOverAll;
    
    % Same as mean but uses median
    farProjectionMedianOverAll=lastPosBeforePred+10000*medianVelPerStep;
    
    [tracksPredictions(i).predictedIntersectionMedianOverAll(1),tracksPredictions(i).predictedIntersectionMedianOverAll(2)]...
        =polyxpoly(airbarPositionX,airbarPositionY,...
        [lastPosBeforePred(1),farProjectionMedianOverAll(1)],[lastPosBeforePred(2),farProjectionMedianOverAll(2)]);
    % Calculate pos error (MedianOverAll)
    tracksPredictions(i).posDeviationMedianOverAll=...
        tracksPredictions(i).predictedIntersectionMedianOverAll - tracksInfoVal(i).trueIntersectionPosGt;
    % Calculate time (MedianOverAll)
    distanceToIntMedianOverAll=norm(lastPosBeforePred-tracksPredictions(i).predictedIntersectionMedianOverAll);
    tracksPredictions(i).predictedTimeMedianOverAll=tracksInfoVal(i).lastIndexBeforePrediction+...
        distanceToIntMedianOverAll/(norm(medianVelPerStep)); % Is already in m / ms
    % Calculate temporal error (MedianOverAll)
    tracksPredictions(i).timeErrorMedianOverAll=...
        tracksInfoVal(i).preciseTimeStepOfIntersection-tracksPredictions(i).predictedTimeMedianOverAll;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate line camera prediction
    switch mainDirection
        case 'x+'
            beltDirection=[1;0];
        case 'x-'
            beltDirection=[-1;0];
        case 'y+'
            beltDirection=[0;1];
        case 'y-'
            beltDirection=[0;-1];
        otherwise
            error('Invalid mainDirection')
    end
    farProjektionLine=lastPosBeforePred+10000*beltDirection;
    
    [tracksPredictions(i).predictedIntersectionLine(1),tracksPredictions(i).predictedIntersectionLine(2)]...
        =polyxpoly(airbarPositionX,airbarPositionY,...
        [lastPosBeforePred(1),farProjektionLine(1)],[lastPosBeforePred(2),farProjektionLine(2)]);
    tracksPredictions(i).posDeviationLine=...
        tracksPredictions(i).predictedIntersectionLine - tracksInfoVal(i).trueIntersectionPosGt;
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate constant velocity with bias correction. We first determine how
% the arrival time changes (using 1% of the tracks). Then, we correct the
% position along the y-coordinate by respecting that the particle now take
% longer/shorter to reach the separation mechanism.
timeErrorCalVel=[tracksPredictions.timeErrorCalVel];
% Use first 1% for correction
temporalCorrection=median(timeErrorCalVel(1:floor(numel(timeErrorCalVel)*0.01)));
predictedTimeConstVelCorr=num2cell([tracksPredictions.predictedTimeCalVel]+temporalCorrection);
% Calculate new time and error
[tracksPredictions.predictedTimeConstVelCorr]=predictedTimeConstVelCorr{:};
timeErrorConstVelCorr=num2cell([tracksInfoVal.preciseTimeStepOfIntersection]-[tracksPredictions.predictedTimeConstVelCorr]);
[tracksPredictions.timeErrorConstVelCorr]=timeErrorConstVelCorr{:};
% Calculate new positions
predictedIntersectionConstVelCorr=[tracksPredictions.predictedIntersectionCalVel];
predictedIntersectionConstVelCorr(2,:)=predictedIntersectionConstVelCorr(2,:)+temporalCorrection*velocitiesCalculated(2,:);
predictedIntersectionConstVelCorrCell=num2cell(predictedIntersectionConstVelCorr,1);
[tracksPredictions.predictedIntersectionConstVelCorr]=predictedIntersectionConstVelCorrCell{:};

posDeviationConstVelCorr=[tracksPredictions.posDeviationCalVel];
posDeviationConstVelCorr(2,:)=posDeviationConstVelCorr(2,:)+temporalCorrection*velocitiesCalculated(2,:);
posDeviationConstVelCorrCell=num2cell(posDeviationConstVelCorr,1);
[tracksPredictions.posDeviationConstVelCorr]=posDeviationConstVelCorrCell{:};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate constant acceleration along y-axis using the time taken
% Generated by the const vel n.c. model

for i=1:size(gt_x_y_vx_vyVal,3)
    timeToSepConstVelCorrXConstAccY=tracksPredictions(i).predictedTimeConstVelCorr-tracksInfoVal(i).lastIndexBeforePrediction;
    % Predict path based on const acc (as above)
    lastPosBeforePred=gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction,i);
    lastVelBeforePred=lastPosBeforePred-gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-1,i);
    beforelastVelBeforePred=gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-1,i)-gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-2,i);
    lastAccBeforePred=lastVelBeforePred-beforelastVelBeforePred;
    

    % Similar to const acc n. c. above
    signChangeAt=lastVelBeforePred(2)/lastAccBeforePred(2);
    if (signChangeAt<0)||(signChangeAt<=timeToSepConstVelCorrXConstAccY)
        % Can simply use formulae for CA
        tracksPredictions(i).predictedIntersectionConstVelCorrXConstAccY=[airbarPositionX(1);...
            lastPosBeforePred(2)+lastVelBeforePred(2)*timeToSepConstVelCorrXConstAccY+0.5*lastAccBeforePred(2)*timeToSepConstVelCorrXConstAccY^2]; % X coorindate is obvious;
    else
        % Position along y-axis only changes until sign change. Can use
        % formula but use signChangeAt instead of timeToSep
        tracksPredictions(i).predictedIntersectionConstVelCorrXConstAccY=...
            [airbarPositionX(1);...
            lastPosBeforePred(2)+lastVelBeforePred(2)*timeToSepConstVelCorrXConstAccY+0.5*lastAccBeforePred(2)*timeToSepConstVelCorrXConstAccY^2]; % X coorindate is obvious
    end
end
%%

posDeviationConstVelCorrXConstAccY=num2cell([tracksInfoVal.trueIntersectionPosGt]-[tracksPredictions.predictedIntersectionConstVelCorrXConstAccY],1);
[tracksPredictions.posDeviationConstVelCorrXConstAccY]=posDeviationConstVelCorrXConstAccY{:};
%%
clear *BeforePred distAlongBelt trueVelBeforePred farProjection*  distanceToInt* timeToSep beforelastVelBeforePred changeAccBeforePred predictedPositions int* allPossibleTimes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate constant velocity with bias correction using additional
% acceleratig component. First obtain optimal acceleations for first 1%
optAccsX=NaN(1,floor(size(gt_x_y_vx_vyVal,3)/100));
optAccsY=NaN(1,floor(size(gt_x_y_vx_vyVal,3)/100));
accRatioX=NaN(1,floor(size(gt_x_y_vx_vyVal,3)/100));
accRatioY=NaN(1,floor(size(gt_x_y_vx_vyVal,3)/100));
for i=1:size(gt_x_y_vx_vyVal,3)/100
    lastPosBeforePred=gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction,i);
    lastVelBeforePred=lastPosBeforePred-gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-1,i);
    beforelastVelBeforePred=gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-1,i)-gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-2,i);
    lastAccBeforePred=lastVelBeforePred-beforelastVelBeforePred;
    
    %Calculate velocity during intersection
    if (floor(tracksInfoVal(i).preciseTimeStepOfIntersection))==tracksInfoVal(i).preciseTimeStepOfIntersection
        xVelDuringIntersection= gt_x_y_vx_vy(1,tracksInfoVal(i).preciseTimeStepOfIntersection+1,i)-gt_x_y_vx_vy(1,tracksInfoVal(i).preciseTimeStepOfIntersection,i);
        yVelDuringIntersection= gt_x_y_vx_vy(2,tracksInfoVal(i).preciseTimeStepOfIntersection+1,i)-gt_x_y_vx_vy(2,tracksInfoVal(i).preciseTimeStepOfIntersection,i);
    else
        xVelDuringIntersection= gt_x_y_vx_vy(1,ceil(tracksInfoVal(i).preciseTimeStepOfIntersection),i)-gt_x_y_vx_vy(1,floor(tracksInfoVal(i).preciseTimeStepOfIntersection),i);
        yVelDuringIntersection= gt_x_y_vx_vy(2,ceil(tracksInfoVal(i).preciseTimeStepOfIntersection),i)-gt_x_y_vx_vy(2,floor(tracksInfoVal(i).preciseTimeStepOfIntersection),i);
    end
    accRatioX(i)=xVelDuringIntersection/lastVelBeforePred(1);
    accRatioY(i)=yVelDuringIntersection/lastVelBeforePred(2);
    
    timeToSepGt=tracksInfoVal(i).preciseTimeStepOfIntersection-tracksInfoVal(i).lastIndexBeforePrediction;
    distAlongBelt=airbarPositionX(1)-lastPosBeforePred(1);
    distOrth=tracksInfoVal(i).trueIntersectionPosGt(2)-lastPosBeforePred(2);
    
    optAccsX(i)=2*(distAlongBelt(1)/timeToSepGt^2-lastVelBeforePred(1)/timeToSepGt);
    optAccsY(i)=2*(distOrth(1)/timeToSepGt^2-lastVelBeforePred(2)/timeToSepGt);
end
avgAccX=median(optAccsX,'omitnan');
% avgAccY=median(optAccsY,'omitnan');
% hist(accRatioY(abs(accRatioY)<=1)),shg
accRatioXMedian=median(accRatioX,'omitnan');
accRatioYMedian=median(accRatioY,'omitnan');

%%
clear *BeforePred distAlongBelt trueVelBeforePred farProjection*  distanceToInt* timeToSep beforelastVelBeforePred changeAccBeforePred predictedPositions int* allPossibleTimes

for i=1:size(gt_x_y_vx_vyVal,3)
    lastPosBeforePred=gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction,i);
    lastVelBeforePred=lastPosBeforePred-gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-1,i);
    beforelastVelBeforePred=gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-1,i)-gt_x_y_vx_vyVal(1:2,tracksInfoVal(i).lastIndexBeforePrediction-2,i);
    lastAccBeforePred=lastVelBeforePred-beforelastVelBeforePred;
    distAlongBelt=airbarPositionX(1)-lastPosBeforePred(1);
    
    
    allPossibleTimesCVIA=roots([0.5*avgAccX,lastVelBeforePred(1),-distAlongBelt]);
    timeToSepCVIA=min(allPossibleTimesCVIA(allPossibleTimesCVIA>=0));
    if imag(timeToSepCA)>0.00001
        error('Obtained imginary value where real value was expected');
    else
        timeToSepCVIA=real(timeToSepCVIA);
    end
    
    allPossibleTimesRatio=roots([lastVelBeforePred(1)+0.5*(lastVelBeforePred(1)*accRatioXMedian-lastVelBeforePred(1)),-distAlongBelt]);
    timeToSepRatio=min(allPossibleTimesRatio(allPossibleTimesRatio>=0));
    tracksPredictions(i).predictedTimeRatio=tracksInfoVal(i).lastIndexBeforePrediction+timeToSepRatio;
        tracksPredictions(i).timeErrorRatio=...
            tracksInfoVal(i).preciseTimeStepOfIntersection-tracksPredictions(i).predictedTimeRatio;
    
    if isempty(timeToSepCVIA)
        warning('Did not find intersection for constant acceleration. Setting to NaN');
    else
        % If found, save time. Everything else will become NaN then by
        % default
        tracksPredictions(i).predictedTimeCVIA=tracksInfoVal(i).lastIndexBeforePrediction+timeToSepCVIA;
        tracksPredictions(i).timeErrorCVIA=...
            tracksInfoVal(i).preciseTimeStepOfIntersection-tracksPredictions(i).predictedTimeCVIA;
    
        % Predict according to CV
        tracksPredictions(i).predictedIntersectionCVIA_CV=...
                [airbarPositionX(1);...
                lastPosBeforePred(2)+lastVelBeforePred(2)*timeToSepCVIA];
        % Predict according to CA
        tracksPredictions(i).predictedIntersectionCVIA_CA=...
                [airbarPositionX(1);...
                lastPosBeforePred(2)+lastVelBeforePred(2)*timeToSepCVIA+0.5*lastAccBeforePred(2)*timeToSepCVIA^2];
        
        tracksPredictions(i).posDeviationCVIA_CV=...
            tracksPredictions(i).predictedIntersectionCVIA_CV - tracksInfoVal(i).trueIntersectionPosGt;
        tracksPredictions(i).posDeviationCVIA_CA=...
            tracksPredictions(i).predictedIntersectionCVIA_CA - tracksInfoVal(i).trueIntersectionPosGt;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Scale acc along y axis identically as for x-axis in the CVIA
        % model. Factor s=avgAccX/lastAccBeforePred(1)
        %
        accScaled=lastAccBeforePred*avgAccX/lastAccBeforePred(1);
        accXScaled=accScaled(2);
        if isnan(accXScaled) % This occurs if lastAcc is zero.
            accXScaled=0;
        end
            
        tracksPredictions(i).predictedIntersectionCVIA_Scale=...
            [airbarPositionX(1);...
            lastPosBeforePred(2)+lastVelBeforePred(2)*timeToSepCVIA+0.5*accXScaled*timeToSepCVIA^2];
        tracksPredictions(i).posDeviationCVIA_Scale=...
            tracksPredictions(i).predictedIntersectionCVIA_Scale - tracksInfoVal(i).trueIntersectionPosGt;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Use acceleration that ensures that a certain percentage of the
        % velocity (e.g., 80%) remains. The ratio of current and future
        % velocity is obtained using a median above
        toAcc=-(1-accRatioYMedian)*lastVelBeforePred(2); % Particle is supposed to accelerate this much
        accRatio=toAcc/timeToSepCVIA; % Get speed by dividing by remaining time
        assert(~isnan(accRatio));
        
        tracksPredictions(i).predictedIntersectionCVIA_Ratio=...
                [airbarPositionX(1);...
                lastPosBeforePred(2)+lastVelBeforePred(2)*timeToSepCVIA+0.5*accRatio*timeToSepCVIA^2];
            
        tracksPredictions(i).posDeviationCVIA_Ratio=...
            tracksPredictions(i).predictedIntersectionCVIA_Ratio - tracksInfoVal(i).trueIntersectionPosGt;
            
    end
      
    
end

%% Assert correct sizes
fn=fieldnames(tracksPredictions);
% assert(numel(fn)==46);
for i=fn'
    currField=[i{:}];
    % Assert non are nan
    assert(~any(any(isnan([tracksPredictions.(currField)]))));
    % Assert correct sizes
    if contains(currField,'predictedIntersection')
        assert(isequal(size([tracksPredictions.(currField)]),[2,size(gt_x_y_vx_vyVal,3)]));
    elseif contains(currField,'posDeviation')
        assert(isequal(size([tracksPredictions.(currField)]),[2,size(gt_x_y_vx_vyVal,3)]));
        % Assert only numerical impcresision along x
        devForAssert=[tracksPredictions.(currField)];
        assert(all((devForAssert(1,:)<eps)));
    elseif contains(currField,'predictedTime')||contains(currField,'timeError')
        assert(isequal(size([tracksPredictions.(currField)]),[1,size(gt_x_y_vx_vyVal,3)]));
    else
        error('Unknown field');
    end
end

