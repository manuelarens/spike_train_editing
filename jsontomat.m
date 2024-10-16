% This script processes EMG data from a gzipped JSON file by:
% 1. Decompressing the file.
% 2. Parsing the JSON content into a MATLAB structure.
% 3. Converting pulse train data from string to numeric arrays.
% 4. Decoding specific JSON fields (e.g., accuracy, signals, IPTS).
% 5. Organizing the data into a `signal` structure with discharge times, raw signals, 
%    and relevant metadata for further analysis.
% 6. Saving the processed data to a .mat file for future use.
%
% Usage: Ensure the gzipped JSON file path is correct. The output is saved
% as a .mat file with the same name as the input JSON.
%
% -----------------------------------------------------------------------------

% Step 1: Decompress the file
filename = 'eigen meting_decomp_edited.json';
gzip_file = filename;  % specify your gzipped JSON file path
out_file = gunzip(gzip_file);             % decompress the gzipped file
json_file = out_file{1};                  % get the path of the decompressed JSON file

% Step 2: Read and parse the JSON file
fid = fopen(json_file, 'r');              % open the decompressed file
raw = fread(fid, inf);                    % read the file content
str = char(raw');                         % convert to string
fclose(fid);                              % close the file

% Step 3: Parse the JSON content
emgfile = jsondecode(str);                % convert JSON to MATLAB structure

yourStruct = emgfile;
numFields = numel(fieldnames(yourStruct));
fieldNames = fieldnames(yourStruct);
for i = 1:numFields
    fieldName = fieldNames{i};
    fieldSize = size(yourStruct.(fieldName));
    fieldType = class(yourStruct.(fieldName));
    fprintf('Field: %s, Size: [%s], Type: %s\n', fieldName, num2str(fieldSize), fieldType);
end
numEntries = numel(yourStruct);

% The input string from emgfile.MUPULSES
input_str = emgfile.MUPULSES;

% Remove the surrounding brackets
input_str = input_str(2:end-1);

% Split the string by '], [' to get individual sublists
split_str = strsplit(input_str, '], [');

% Initialize the cell array to store the result
cell_array = cell(1, length(split_str));

% Process each split string (each sublist)
for i = 1:length(split_str)
    % Remove any remaining brackets and convert the string to a numeric array
    clean_str = strrep(split_str{i}, '[', '');
    clean_str = strrep(clean_str, ']', '');
    
    % Convert to numeric array
    num_array = str2num(clean_str); %#ok<ST2NM> 
    
    % Store in the cell array
    cell_array{i} = num_array;
end

% Now cell_array contains the numeric arrays
emgfile.MUPULSES = cell_array;

emgfile.ACCURACY = jsondecode(emgfile.ACCURACY);
emgfile.ACCURACY = emgfile.ACCURACY.data;

emgfile.RAW_SIGNAL = jsondecode(emgfile.RAW_SIGNAL);
emgfile.RAW_SIGNAL = emgfile.RAW_SIGNAL.data;

emgfile.REF_SIGNAL = jsondecode(emgfile.REF_SIGNAL);
emgfile.REF_SIGNAL = emgfile.REF_SIGNAL.data;

emgfile.EXTRAS = jsondecode(emgfile.EXTRAS);
emgfile.EXTRAS = emgfile.EXTRAS.data;

emgfile.IPTS = jsondecode(emgfile.IPTS);
emgfile.IPTS = emgfile.IPTS.data;

emgfile.BINARY_MUS_FIRING = jsondecode(emgfile.BINARY_MUS_FIRING);
emgfile.BINARY_MUS_FIRING = emgfile.BINARY_MUS_FIRING.data;

emgfile.SOURCE = strrep(emgfile.SOURCE, '"', '');
emgfile.FILENAME = strrep(emgfile.FILENAME, '"', '');

emgfile.FSAMP = str2double(emgfile.FSAMP);
emgfile.IED = str2double(emgfile.IED);
emgfile.EMG_LENGTH = str2double(emgfile.EMG_LENGTH);
emgfile.NUMBER_OF_MUS = str2double(emgfile.NUMBER_OF_MUS);

signal.data = emgfile.RAW_SIGNAL';
signal.fsamp = emgfile.FSAMP;
[~, signal.nChan] = size(emgfile.RAW_SIGNAL);
signal.ngrid = 1;
signal.gridname = {'4-8-L'};  % A cell array with one grid name
signal.muscle = {'TA'};  % A cell array with one muscle name
signal.target = emgfile.REF_SIGNAL';
signal.path = emgfile.REF_SIGNAL';

signal.Pulsetrain = {};
signal.Pulsetrain{1,1} = emgfile.IPTS;
signal.Pulsetrain{1,1} = signal.Pulsetrain{1,1}';

signal.Dischargetimes = emgfile.MUPULSES';
[a,b] = size(signal.Dischargetimes);
signal.Dischargetimes = mat2cell(signal.Dischargetimes, a, ones(1,b));

signal.Dischargetimes = signal.Dischargetimes{1,1};
for i = 1:length(signal.Dischargetimes)
    % Convert the cell content to an array, increment by 1, and store it back
    signal.Dischargetimes{i} = signal.Dischargetimes{i} + 1;
end

signal.Dischargetimes = signal.Dischargetimes';

signal.EMGmask = {}; % Initialize cell array with ngrid cells
signal.EMGmask{1} = zeros(size(signal.data,1), signal.ngrid);

signal.emgtype = 1;

edition = signal;
save(strrep(filename, '.json', '.mat'), 'signal', 'edition');
clearvars -except emgfile edition signal