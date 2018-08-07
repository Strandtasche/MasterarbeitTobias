% @author Florian Pfaff pfaff@kit.edu
% @date 2016

% Read x, y, v_x, v_y and ID
format='x_y_vx_vy';
% format='x_y_z_vx_vy_vz';
if strcmp(format,'x_y_vx_vy')
    formatSpec = '%f %f %*f %f %f %*f %*f %*f %*f %*f %*f %*f %*f %*f %f %*f\n'; % For x y vx vy
elseif strcmp(format,'x_y_z_vx_vy_vz')
    formatSpec = '%f %f %f %f %f %f %*f %*f %*f %*f %*f %*f %*f %*f %f %*f\n'; % For x y vx vy
else
    error('Format not supported');
end
% path = 'sphresFirctionlow\MATP_Kugeln_40_200_Friction_low\';
% path='C:\Users\emper\Documents\tmpdata\MATP_BaseCase_Kugeln_200_40';
% path='D:\TrackSortData\Chips_115\MATP_chips_115'
path='D:\TrackSortData\Cylinders_115\MATP'
files = dir(fullfile(path,'t0*.txt'));
if isempty(files)
    error('Folder does not contain text files of the desired format.')
end

% Get number of particles by parsing first file
filename = files(1).name;
fileID = fopen(fullfile(path,filename),'r');
tmp = fscanf(fileID,formatSpec,[5,inf]);
fclose(fileID);
if size(tmp,2)<10
    error('Less than 10 particles in data set. This is definitely wrong');
end
% Initialize 3D matrix: coordinates-timestep-track
if strcmp(format,'x_y_vx_vy')
    gt_x_y_vx_vy=NaN(4,numel(files),size(tmp,2));
elseif strcmp(format,'x_y_z_vx_vy_vz')
    gt_x_y_z_vx_vy_vz=NaN(6,numel(files),size(tmp,2));
end

for j = 1:length(files)
    if mod(j,50)==0
        fprintf('Reading file %d of %d\n',j,length(files));
    end
    filename = files(j).name;
    fileID = fopen(fullfile(path,filename),'r');
    if strcmp(format,'x_y_vx_vy')
        tmp=fscanf(fileID,formatSpec,[5,inf]);
    elseif strcmp(format,'x_y_z_vx_vy_vz')
        tmp=fscanf(fileID,formatSpec,[7,inf]);
    end
    
    fclose(fileID);
    if isempty(tmp)
        continue;
    end
    if strcmp(format,'x_y_vx_vy')
        gt_x_y_vx_vy(:,j,tmp(5,:))=tmp(1:4,:);
    elseif strcmp(format,'x_y_z_vx_vy_vz')
        gt_x_y_z_vx_vy_vz(:,j,tmp(7,:))=tmp(1:6,:);
    end
    
    
end
%%
if strcmp(format,'x_y_vx_vy')
    save('gt2.mat','-v7.3','gt_x_y_vx_vy')
elseif strcmp(format,'x_y_z_vx_vy_vz')
    save('gt2.mat','-v7.3','gt_x_y_z_vx_vy_vz')
end
