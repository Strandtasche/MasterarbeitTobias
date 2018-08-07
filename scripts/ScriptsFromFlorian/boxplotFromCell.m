function boxplotFromCell(datacell,labelcell,varargin)
    % Varargin can be used to pass on additional arguments
    % v1.0 by Florian Pfaff
    numelLargest=max(cellfun(@numel,datacell));
    boxplotMat=NaN(numelLargest,numel(datacell));
    for i=1:numel(datacell)
        assert(~isempty(datacell{i}));
        boxplotMat(1:numel(datacell{i}),i)=datacell{i};
    end
    if nargin>=2
        boxplot(boxplotMat,labelcell,varargin{:});
    else
        boxplot(boxplotMat);
    end
end