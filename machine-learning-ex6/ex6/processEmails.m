function words = processEmails(path, filenames, words)
% PROCESSEMAILS takes in email contents and an existing list of words,
% performs replacement for digits, URLs, web addresses, and dollar signs,
% and updates an existing tally of words used to form a vocabulary list. 
dir = pwd;
cd([pwd, path]);

n = size(filenames,1);
for i = 3:(n-1)
    disp(i);
    % Load File
    fid = fopen(filenames(i,:));
    if fid
        email_contents = fscanf(fid, '%c', inf);
        fclose(fid);
    else
        email_contents = '';
        fprintf('Unable to open %s\n', filename);
    end
    % Find the Headers ( \n\n and remove )
    % Uncomment the following lines if you are working with raw emails with the
    % full headers
    cd(dir);
    
    hdrstart = strfind(email_contents, ([char(10) char(10)]));
    email_contents = email_contents(hdrstart(1):end);

    % Lower case
    email_contents = lower(email_contents);

    % Strip all HTML
    % Looks for any expression that starts with < and ends with > and replace
    % and does not have any < or > in the tag it with a space
    email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

    % Handle Numbers
    % Look for one or more characters between 0-9
    email_contents = regexprep(email_contents, '[0-9]+', 'number');

    % Handle URLS
    % Look for strings starting with http:// or https://
    email_contents = regexprep(email_contents, ...
                               '(http|https)://[^\s]*', 'httpaddr');

    % Handle Email Addresses
    % Look for strings with @ in the middle
    email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

    % Handle $ sign
    email_contents = regexprep(email_contents, '[$]+', 'dollar');

    % ========================== Tokenize Email ===========================

    % Process file

    found = 0;
    while ~isempty(email_contents)

        % Tokenize and also get rid of any punctuation
        [str, email_contents] = ...
           strtok(email_contents, ...
                  [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);

        % Remove any non alphanumeric characters
        str = regexprep(str, '[^a-zA-Z0-9]', '');

        % Stem the word 
        % (the porterStemmer sometimes has issues, so we use a try catch block)
        try str = porterStemmer(strtrim(str)); 
        catch str = ''; continue;
        end;

        % Skip the word if it is too short
        if length(str) < 1
           continue;
        end

        num_words = size(words,1);
        for j = 1:num_words
            if strcmp(str, words{j,1})
                found = 1;
                words{j,2} = words{j,2} + 1;
                break;
            end
        end
        if ~found
            words{num_words + 1,1} = str;
            words{num_words + 1,2} = 1;
        end
        found = 0;
    end
    cd([pwd, path]);
end
cd(dir);
end