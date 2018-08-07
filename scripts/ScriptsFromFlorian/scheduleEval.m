% @author Florian Pfaff pfaff@kit.edu
% @date 2016
evalConfig=struct('predictionEdgePosition',0.638,'airbarPosition',0.788,...
    'gtFile','spheres_other_friction_x_y_vx_vy.mat','samplingRateImData',1000); %#ok<*NASGU>
evaluation
%%
evalConfig=struct('predictionEdgePosition',0.638,'airbarPosition',0.788,...
    'gtFile','groundtruthCylinders.mat','samplingRateImData',1000); %#ok<*NASGU>
evaluation
%%
evalConfig=struct('predictionEdgePosition',0.638,'airbarPosition',0.788,...
    'gtFile','groundtruthPlates.mat','samplingRateImData',1000);
evaluation
%%
evalConfig=struct('predictionEdgePosition',0.638,'airbarPosition',0.788,...
    'gtFile','groundtruthSpheres.mat','samplingRateImData',1000);
evaluation

evalConfig=struct('predictionEdgePosition',0.638,'airbarPosition',0.788,...
    'gtFile','spheres_other_friction_x_y_vx_vy.mat','samplingRateImData',200); %#ok<*NASGU>
evaluation
evalConfig=struct('predictionEdgePosition',0.638,'airbarPosition',0.788,...
    'gtFile','groundtruthCylinders.mat','samplingRateImData',200); %#ok<*NASGU>
evaluation
evalConfig=struct('predictionEdgePosition',0.638,'airbarPosition',0.788,...
    'gtFile','groundtruthPlates.mat','samplingRateImData',200);
evaluation
evalConfig=struct('predictionEdgePosition',0.638,'airbarPosition',0.788,...
    'gtFile','groundtruthSpheres.mat','samplingRateImData',200);
evaluation
