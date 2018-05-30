
function trackHistoryUpdated = testSuite(varargin)
%testSuite: check if
%   Detailed explanation goes here
    assert((nargin == 2) ||(nargin == 1), 'Wrong number of arguments')
    
    %check if first argument is filename or object and load accordingly
    if isa(varargin{1}, 'char') || isstring(varargin{1})
        load(varargin{1});
    else
        trackHistory_dummy = varargin{1};
    end
    
    %trackHistoryUpdated = 0;
    
    if nargin == 1
        threshold = 5;
    else
        threshold = varargin{2};
    end
    
    ok = 1;
    %Removal reverse order
    removal = [];
    %iterate over all Tracks in Trackhistory
    for it = 1:size(trackHistory_dummy,2)
        %calculate number of non-NaN
        nanified = isnan(transpose(trackHistory_dummy(it).RawMeasurements));
        invertedNani = sum(arrayfun(@(x) 1-x, nanified));
        %disp(invertedNani(1))
        if invertedNani(1) < threshold
            %disp(strcat("not enough values in track no. ", num2str(it)))
            ok = 0;
            removal = [it, removal];
        end
    end

    %removal of tracks with too many NaN Values
    for rem = removal
        trackHistory_dummy(rem) = [];
    end
        
    if ok == 1
        disp("All correct!")
    end
    
    trackHistoryUpdated = trackHistory_dummy;
    %disp(trackHistoryUpdated)
end


%disp(trackHistory(1).RawMeasurements)


