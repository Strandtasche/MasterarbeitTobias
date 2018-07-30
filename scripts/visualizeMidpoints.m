% sent to me on 2018-07-25 by Florian

% before running set borders=[beltBordersX;beltBordersY]

function visualizeMidpoints(midpointMatrix,borders)
if nargin==0
    midpointMatrix=evalin('base','midpointMatrix');
end
if nargin<2
    borders=evalin('base','borders');
end
shg,clf,hold on
axis(reshape(borders',1,[]));
for i=1:size(midpointMatrix,3)
    valid=~isnan(midpointMatrix(1,:,i));
    if ~any(valid)
        continue
    end
    cla;
    scatter(midpointMatrix(1,valid,i),midpointMatrix(2,valid,i));
    drawnow
end
end
