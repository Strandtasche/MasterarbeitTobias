% @author Florian Pfaff pfaff@kit.edu
% @date 2016
xymin=min(min(gt_x_y_vx_vy(1:2,:,:),[],2),[],3);
xymax=max(max(gt_x_y_vx_vy(1:2,:,:),[],2),[],3);


% Specify if particles are traveling mainly along x or y axis, airbar is
% assumed to be orthogonal. Movement direction x+ means that it moves in
% positive x direction, x- that it moves in negative - direction

if strncmp(mainDirection,'x',1) % Extend of airbar is automatically generated from orientation and position
    predictionEdgePositionX=[predictionEdgePosition,predictionEdgePosition];
    predictionEdgePositionY=[xymin(2)-1000,xymax(2)+1000]; % Make the edges larger so they can also be used for inaccurate predictions
    airbarPositionX=[airbarPosition,airbarPosition];
    airbarPositionY=[xymin(2)-1000,xymax(2)+1000];
elseif strncmp(mainDirection,'y',1)
    predictionEdgePositionX=[xymin(1)-1000,xymax(1)+1000];
    predictionEdgePositionY=[predictionEdgePosition,predictionEdgePosition];
    airbarPositionX=[xymin(1)-1000,xymax(1)+1000];
    airbarPositionY=[airbarPosition,airbarPosition];
else
    error('Invalid orientation');
end
%%
% Visualize edge and airbar
figure(1),clf, hold on
axis([xymin(1),xymax(1),xymin(2),xymax(2)])
plot(predictionEdgePositionX,predictionEdgePositionY) % If not visible: is it outside the area of the axis?
plot(airbarPositionX,airbarPositionY)

drawnow
%%
tracksInfo=repmat(struct('trueIntersectionPosGt',NaN(2,1),'preciseTimeStepOfIntersection',NaN,'lastIndexBeforePrediction',NaN),1,size(gt_x_y_vx_vy,3));
for i=1:size(gt_x_y_vx_vy,3)
    if mod(i,50)==0,fprintf('Getting true Intersection of track %d of %d\n',i,size(gt_x_y_vx_vy,3)),end
    % Find intersection with airbar
    [xIntAb,yIntAb]=polyxpoly(airbarPositionX,airbarPositionY,...
                                    gt_x_y_vx_vy(1,:,i),gt_x_y_vx_vy(2,:,i));
    if isempty(xIntAb)&&(strcmp(gtFileCurr,'groundtruthSpheres.mat')...
            &&ismember(i,[1112,1974,2069,2766,3246,3272,3413,3472,3479,3537,3606,3705,3977,3991,4070,4209])...
            || strcmp(gtFileCurr,'groundtruthPlates.mat')...
            &&ismember(i,[72,113,119,178,186,269,305])) % Gibt noch deutlich mehr!
        warning('This track is known to be faulty, skipping.');
        continue
    elseif isempty(xIntAb)
        warning('No intersection with the airbar was detected for this ground truth track. While this track is not known to be faulty, it is very likely');
        continue
    elseif numel(xIntAb)>1
        warning(...,
            'Multiple intersections found. This should not happen for groundtruth, but can happen for data sets in which the particles return to zero.')
        xIntAb=xIntAb(1);
        yIntAb=yIntAb(1);
    end
    tracksInfo(i).trueIntersectionPosGt(1)=xIntAb;
    tracksInfo(i).trueIntersectionPosGt(2)=yIntAb;
    % Find last point before prediction to generate a prediction using the
    % DEM simulation. Also find point before the airbar to calculate the time at which it intersects the airbar)
    switch mainDirection
        case 'x+'
            lastIndBeforePred=find(gt_x_y_vx_vy(1,:,i)<predictionEdgePosition,1,'last');
            lastIndBeforeAirbar=find(gt_x_y_vx_vy(1,:,i)<airbarPosition,1,'last');
        case 'x-'
            lastIndBeforePred=find(gt_x_y_vx_vy(1,:,i)>predictionEdgePosition,1,'last');
            lastIndBeforeAirbar=find(gt_x_y_vx_vy(1,:,i)>airbarPosition,1,'last');
        case 'y+'
            lastIndBeforePred=find(gt_x_y_vx_vy(2,:,i)<predictionEdgePosition,1,'last');
            lastIndBeforeAirbar=find(gt_x_y_vx_vy(2,:,i)<airbarPosition,1,'last');
        case 'y-'
            lastIndBeforePred=find(gt_x_y_vx_vy(2,:,i)>predictionEdgePosition,1,'last');
            lastIndBeforeAirbar=find(gt_x_y_vx_vy(2,:,i)<airbarPosition,1,'last');
        otherwise
            error('Invalid mainDirection')
    end
    tracksInfo(i).lastIndexBeforePrediction=lastIndBeforePred;
    tracksInfo(i).lastVelocityBeforePrediction(1)=gt_x_y_vx_vy(3,lastIndBeforePred,i);
    tracksInfo(i).lastVelocityBeforePrediction(2)=gt_x_y_vx_vy(4,lastIndBeforePred,i);
    % Find time at which it arrives at airbar
    % First, find distance between last point before and last point after
    distanceTraveled=norm(gt_x_y_vx_vy(1:2,lastIndBeforeAirbar,i)-gt_x_y_vx_vy(1:2,lastIndBeforeAirbar+1,i));
    distanceToIntersection=norm(gt_x_y_vx_vy(1:2,lastIndBeforeAirbar,i)-tracksInfo(i).trueIntersectionPosGt);
    assert(distanceTraveled>distanceToIntersection);
    tracksInfo(i).preciseTimeStepOfIntersection=lastIndBeforeAirbar+distanceToIntersection/distanceTraveled;
    
    
end


save('gtTested.mat','tracksInfo','mainDirection','airbarPosition','airbarPositionX',...
    'airbarPositionY','predictionEdgePosition','predictionEdgePositionX','predictionEdgePositionY');