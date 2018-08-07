% @author Florian Pfaff pfaff@kit.edu
% @date 2016

beltBordersX=[0.388,0.788];
% beltBordersX=[0.388,0.788+0.05];
beltBordersY=[0,0.18];
% return

% % % %% Show velocities using quiver
% % % skip=1;
% % % for ind=1:size(gt_x_y_vx_vy,3)
% % %     onBelt=(gt_x_y_vx_vy(1,:,ind)>beltBordersX(1))&(gt_x_y_vx_vy(1,:,ind)<beltBordersX(2))...
% % %         &(gt_x_y_vx_vy(2,:,ind)>beltBordersY(1))&(gt_x_y_vx_vy(2,:,ind)<beltBordersY(2));
% % % 
% % %     xyOnBelt=gt_x_y_vx_vy(:,onBelt,ind);
% % %     quiver(xyOnBelt(1,1:skip:end),xyOnBelt(2,1:skip:end),skip*xyOnBelt(3,1:skip:end),skip*xyOnBelt(4,1:skip:end))
% % %     xlim(beltBordersX)
% % %     ylim(beltBordersY)
% % %     drawnow
% % % end
%% Show velocity or acceleration
% axis1mode='posx';
noParticle=15;
axis1mode='posx';
axis2mode='vely'; % Use vel or acc or jerk
randomize=true;
switch axis1mode
    case 'posx'
        axis1index=1;
    case 'posy'
        axis1index=2;
    case 'velx'
        axis1index=3;
    case 'vely'
        axis1index=4;
    case 'time'
        axis1index=NaN;
end

% xdata='velx';
% ydata=
if strcmp(axis1mode,'time')
    xlabel('Time in time steps on the belt');
else
    xlabel('Position on the belt along the transport direction in m');
end
switch axis2mode
    case {'velx','vely'}
        ylabel('Velocity in m / s');
    case {'accx','accy'}
        ylabel('Acceleration in m / s^2');
    case {'jerkx','jerky'}
        ylabel('Jerk in m / s^3');
end
assert(noParticle<size(gt_x_y_vx_vy,3));
randOrder=randperm(size(gt_x_y_vx_vy,3));
cla, hold on
for ind=randOrder(1:15)
%     if mod(ind,noParticle)==1 % To see no more than 15 tracks at once
%         if ind>1
%             return;
%         end
%         cla,hold on
%     end
    onBelt=(gt_x_y_vx_vy(1,:,ind)>beltBordersX(1))&(gt_x_y_vx_vy(1,:,ind)<beltBordersX(2))...
        &(gt_x_y_vx_vy(2,:,ind)>beltBordersY(1))&(gt_x_y_vx_vy(2,:,ind)<beltBordersY(2));
    if sum(abs(diff(onBelt)))>2
        warning('Particle has temporarily left belt')
    end
    xyOnBelt=gt_x_y_vx_vy(:,onBelt,ind);
    if strcmp(axis1mode,'time')
        xVals=1:size(xyOnBelt,2);
    else
        xVals=xyOnBelt(axis1index,1:end);
    end
    switch axis2mode
        case 'velx'
            plot(xVals,xyOnBelt(3,:))
        case 'vely'
            plot(xVals,xyOnBelt(4,:))
        case 'accx'
            plot(xVals(1:end-1),1000*diff(xyOnBelt(3,:),1))
        case 'accy'
            plot(xVals(1:end-1),1000*diff(xyOnBelt(4,:),1))
        case 'jerkx'
            plot(xVals(1:end-2),1000^2*diff(xyOnBelt(3,:),2))
        case 'jerky'
            plot(xVals(1:end-2),1000^2*diff(xyOnBelt(4,:),2))
        otherwise
            error('Not recognized');
    end
    
    drawnow
%     ylim([-0.003,0.003]);
    ylim([-0.1,0.33])
    xlim([0.55,0.78])
%     ylim([0.2,1.7]);
    pause(0.1) 
end

return
%% Interpolate
% load groundtruthSpheres.mat % replaced by other data set!
load spheres_other_friction_x_y_vx_vy.mat
gt_x_y_vx_vySph=gt_x_y_vx_vy;
load groundtruthPlates.mat
gt_x_y_vx_vyPla=gt_x_y_vx_vy;
load groundtruthCylinders.mat
gt_x_y_vx_vyCyl=gt_x_y_vx_vy;

%%
%}
% plot([-100,100],predictionEdgePosition);
x=beltBordersX(1):0.001:beltBordersX(2);
xlim(beltBordersX)
xlabel('Position on the belt along the transport direction in m');
axis2mode='vely';
switch axis2mode
    case {'velx','vely'}
        ylabel('Velocity in m / s');
    case 'acc'
        ylabel('Acceleration in m / s^2');
    case 'jerk'
        ylabel('Jerk in m / s^3');
end
onBeltInterpAll=NaN(size(gt_x_y_vx_vy,3),numel(x));
onlyPlotMean=true;
interpolationMethod='spline';
for ind=1:size(gt_x_y_vx_vy,3)
    if ~onlyPlotMean && (mod(ind,50)==1)
        cla;hold on
    end
    onBelt=(gt_x_y_vx_vy(1,:,ind)>beltBordersX(1))&(gt_x_y_vx_vy(1,:,ind)<beltBordersX(2))...
        &(gt_x_y_vx_vy(2,:,ind)>beltBordersY(1))&(gt_x_y_vx_vy(2,:,ind)<beltBordersY(2));

    xyCloseToBelt=gt_x_y_vx_vy(:,onBelt,ind);
    
    switch axis2mode
        case 'velx'
            toInterpolateX=xyCloseToBelt(1,:);
            toInterpolateY=xyCloseToBelt(3,:);
        case 'vely'
            toInterpolateX=xyCloseToBelt(1,:);
            toInterpolateY=xyCloseToBelt(4,:);
        case 'acc'
            toInterpolateX=xyCloseToBelt(1,1:end-1);
            toInterpolateY=1000*diff(xyCloseToBelt(3,:),1);
        case 'jerk'
            toInterpolateX=xyCloseToBelt(1,1:end-2);
            toInterpolateY=1000^2*diff(xyCloseToBelt(3,:),2);
        otherwise
            error('Not recognized');
    end
    
    if numel(toInterpolateX)<2
        onBeltInterpAll(ind,:)=missing;
        continue
    else
        switch interpolationMethod
            case 'linear'
                interpol=interp1(toInterpolateX,toInterpolateY,x);
            case 'spline'
                interpol=spline(toInterpolateX,toInterpolateY,x);
            otherwise
                error('Not recognized');
        end
        onBeltInterpAll(ind,:)=interpol;
    end
% %     
    if ~onlyPlotMean
        plot(x(2:end-1),interpol(2:end-1));
        drawnow
        pause(0.1)
        
        xlim([0.55,0.78])
        ylim([0.2,1.7]);
    end
end
%%
switch size(gt_x_y_vx_vy,3)
    case 3713
        color=[0    0.4470    0.7410];
    case 4427
        color=[0.8500    0.3250    0.0980];
    case 4357
        color=[0.9290    0.6940    0.1250];
    otherwise
        color='r';
end
plot(x,mean(onBeltInterpAll,1,'omitnan'),'color',color),hold on
% plot(x,median(onBeltInterpAll,1,'omitnan')),hold on
h=plot(x,mean(onBeltInterpAll,1,'omitnan')-std(onBeltInterpAll,1,'omitnan'),'--','color',color);
hasbehavior(h,'legend',false);
h=plot(x,mean(onBeltInterpAll,1,'omitnan')+std(onBeltInterpAll,1,'omitnan'),'--','color',color);
hasbehavior(h,'legend',false);
return

%%
% open meanVel3.fig
open meanVelNewSpheres.fig
legend('Spheres','Cylinders','Cuboids','Location','Southeast')
xlim([0.55,beltBordersX(2)])
xlabel('Position on the belt along the transport direction in m');
ylabel('Velocity in m\,/\,s');
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\VelocitiesNewTrunc.pdf
%%
% open medianAcc.fig
% open medianAccNewSpheres.fig
legend('Spheres','Cylinders','Cuboids','Location','Southwest')
xlim([0.55,beltBordersX(2)])
ylim([-0.08,1.8])
xlabel('Position on the belt along the transport direction in m');
ylabel('Acceleration in m\,/\,s$^2$');
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\AccelerationsNewTrunc.pdf
% open meanVel3.fig
%%
open meanStdVel.fig
legend('Spheres','Cylinders','Cuboids','Location','Southeast')
set(findobj(gca,'LineStyle','--'),'LineStyle',':')
xlim([0.55,beltBordersX(2)])
xlabel('Position on the belt along the transport direction in m');
ylabel('Velocity in m\,/\,s');
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\Draft\VelocitiesMeanAndStdTrunc.pdf
%%
% open medianAcc.fig
% open medianAccNewSpheres.fig
open meanStdAcc.fig
legend('Spheres','Cylinders','Cuboids','Location','Southwest')
xlim([0.55,beltBordersX(2)])
ylim([-0.5,2.8])
xlabel('Position on the belt along the transport direction in m');
ylabel('Acceleration in m\,/\,s$^2$');
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\AccelerationsAndStdTrunc.pdf
%% with dotted lines
open meanStdAcc.fig
legend('Spheres','Cylinders','Cuboids','Location','Southwest')
xlim([0.55,beltBordersX(2)])
set(findobj(gca,'LineStyle','--'),'LineStyle',':')
ylim([-0.5,3.2])
xlabel('Position on the belt along the transport direction in m');
ylabel('Acceleration in m\,/\,s$^2$');
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\Draft\AccelerationsMeanAndStdTrunc.pdf

%%
% open 50Vely.fig
open 30VelyCuboids.fig
xlim([0.55,beltBordersX(2)])
xlabel('Position on the belt along the transport direction in m');
ylabel('Velocity in m\,/\,s')
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\Draft\Velocitiesy30CuboidsTrunc.pdf

%%
open 25VelyCuboids.fig
xlim([0.55,beltBordersX(2)])
xlabel('Position on the belt along the transport direction in m');
ylabel('Velocity in m\,/\,s')
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\Draft\Velocitiesy25CuboidsTrunc.pdf
%%
open 20AccyCuboids.fig
xlim([0.55,beltBordersX(2)])
xlabel('Position on the belt along the transport direction in m');
ylabel('Acceleration in m\,/\,s$^2$');
ylim([-0.31,0.31])
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\Draft\Accelerationsy20CuboidsTrunc.pdf
%%
open 15AccyCuboids.fig
xlim([0.55,beltBordersX(2)])
xlabel('Position on the belt along the transport direction in m');
ylabel('Acceleration in m\,/\,s$^2$');
% ylim([-0.31,0.31])
prepareFig([9,5])
export_fig -transparent ..\..\TII_Pfaff-TrackSortImprovedModels\Draft\Accelerationsy15CuboidsTrunc.pdf