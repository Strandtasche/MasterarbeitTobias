% sent to me on 2018-07-25 by Florian

Bool_Pos_z = false;
% downsampleBy=1;
% startAtIndex=1;
numFrames=size(pos,2); % use all Frames
% numFrames=19999;
beltBordersX=[0.388,0.788];
beltBordersY=[0,0.18];
% 
% currentlyOnBelt=squeeze(...
%     pos(1,startAtIndex:downsampleBy:startAtIndex+numFrames-1,:)>beltBordersX(1) ...
%     & pos(1,startAtIndex:downsampleBy:startAtIndex+numFrames-1,:)<beltBordersX(2)...
%     & pos(2,startAtIndex:downsampleBy:startAtIndex+numFrames-1,:)>beltBordersY(1)  ...
%     & pos(2,startAtIndex:downsampleBy:startAtIndex+numFrames-1,:)<beltBordersY(2));
% 
currentlyOnBelt=squeeze(...
    pos(1,:,:)>beltBordersX(1) ...
    & pos(1,:,:)<beltBordersX(2)...
    & pos(2,:,:)>beltBordersY(1)  ...
    & pos(2,:,:)<beltBordersY(2));

%%
% Eliminate particles flying weirdly
if Bool_Pos_z
    for track=1:size(pos,3)
        if max(pos(3,currentlyOnBelt(:,track),track))>0.01
            currentlyOnBelt(:,track)=false;
        end
    end
end
%%
numberOfMidpoints=sum(currentlyOnBelt,2);
midpointMatrix=NaN(2,max(numberOfMidpoints),numel(numberOfMidpoints));
orientationMatrix=NaN(max(numberOfMidpoints),numel(numberOfMidpoints));
for t=1:numel(numberOfMidpoints)
    midpointMatrix(:,1:numberOfMidpoints(t),t)=pos(1:2,t,currentlyOnBelt(t,:));
    if Bool_Pos_z
        orientationMatrix(1:numberOfMidpoints(t),t)=rotz(t,currentlyOnBelt(t,:));
    end
    % Assert that everything we have saved is valid
    assert(~any(any(any(isnan(midpointMatrix(:,1:numberOfMidpoints(t),t))))));
    if Bool_Pos_z
        assert(~any(any(isnan(orientationMatrix(1:numberOfMidpoints(t),t)))));
    end
end
midpointToGtMapping=NaN(max(numberOfMidpoints),numel(numberOfMidpoints));
for t=1:numel(numberOfMidpoints)
    midpointIDs=find(currentlyOnBelt(t,:));
    midpointToGtMapping(1:numel(midpointIDs),t)=midpointIDs;
end

assert(isequal(isnan(midpointToGtMapping),isnan(squeeze(midpointMatrix(1,:,:)))));
% save cuboids1.15WithOrientationWithGT.mat numberOfMidpoints midpointMatrix midpointToGtMapping orientationMatrix
save cylinders1.15WithOrientationWithGT.mat numberOfMidpoints midpointMatrix midpointToGtMapping orientationMatrix

midpointMatrixNoiseFree=midpointMatrix;
% for noiseStandardDeviation=[0.0001,0.0003,0.0005,0.0008,0.001]
%     % NOISES ARE STANDARD DEVATIONS. VARIANCE IS SQUARE THEREOF!!!!
%     midpointMatrix=addPosNoise(midpointMatrixNoiseFree,noiseStandardDeviation);
%     % Truncate to belt borders
%     tooLowX=find(beltBordersX(1)>midpointMatrix(1,:));
%     midpointMatrix(1,tooLowX)=beltBordersX(1);
%     tooHighX=find(beltBordersX(2)<midpointMatrix(1,:));
%     midpointMatrix(1,tooHighX)=beltBordersX(2);
%     
%     tooLowY=find(beltBordersY(1)>midpointMatrix(2,:));
%     midpointMatrix(2,tooLowY)=beltBordersY(1);
%     tooHighY=find(beltBordersY(2)<midpointMatrix(2,:));
%     midpointMatrix(2,tooHighY)=beltBordersY(2);
% %     save(sprintf('cuboids1.15WithOrientationWithGT%5.5Gtrunc.mat',noiseStandardDeviation),'numberOfMidpoints','midpointMatrix','midpointToGtMapping','orientationMatrix');
%     save(sprintf('cylinders1.15WithOrientationWithGT%5.5Gtrunc.mat',noiseStandardDeviation),'numberOfMidpoints','midpointMatrix','midpointToGtMapping','orientationMatrix');
% end
