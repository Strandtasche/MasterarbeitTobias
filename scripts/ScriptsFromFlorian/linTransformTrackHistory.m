% @author Florian Pfaff pfaff@kit.edu
% @date 2016
function trackHistory=linTransformTrackHistory(trackHistory,A,t)
    if nargin==2
        t=[0;0];
    end
    for i=1:length(trackHistory)
        trackHistory(i).Posterior=A*trackHistory(i).Posterior+repmat(t,1,size(trackHistory(i).Posterior,2));
        trackHistory(i).Tracks_measurment=A*trackHistory(i).Tracks_measurment+...
            repmat(t,1,size(trackHistory(i).Tracks_measurment,2));
    end
end