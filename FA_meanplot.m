%{ 
FA dataset processing and plotting table-collected data
Shruthi Srinivasan
Duke REU 2021
%}

clear
clc
close all

subjects = ["S00393", "S00490", "S00613", "S00680", "S00699", "S00795", "S01952"];
%subjects = ["S00393"];
for i = 1:length(subjects)
    pathl = (fullfile(subjects(i) + '_l.csv'));
    pathr = (fullfile(subjects(i) + '_r.csv'));

    fa_data_l = csvread(pathl);
    fa_data_r = csvread(pathr);

    l_data = fa_data_l(2:length(fa_data_l)-1, :);
    fprintf('Size of FA Data L is %1.4f\n', size(l_data, 1));
    r_data = fa_data_r(2:length(fa_data_r)-1, :);
    fprintf('Size of FA Data R is %1.4f\n', size(r_data, 1));
    
    [rows, cols] = size(l_data);

    %x?± z* ?/?n
    % LEFT SIDE
    mean_l = fa_data_l(length(fa_data_l),:);
    lsigma = zeros(1, cols);

    for j = 1:cols
        a = l_data(1:length(l_data)-1, j);
        a_stdev = std(a);
        lsigma(j) = a_stdev; 
    end
    
    z = 1.96;
    n = 60;
    ci_margin_l = z.*(lsigma./(sqrt(n)));
    
    % RIGHT SIDE
    mean_r = fa_data_r(length(fa_data_r),:);
    rsigma = zeros(1, cols);

    for k = 1:cols
        a = r_data(1:length(r_data)-1, k);
        a_stdev = std(a);
        rsigma(k) = a_stdev; 
    end
    
    z = 1.96;
    n = 60;
    ci_margin_r = z.*((rsigma)./(sqrt(n)));
    
    streamlines = 1:60;

    %PLOTTING
    figure()
    strm = [streamlines, fliplr(streamlines)];
    lci_plot = [mean_l+ci_margin_l, fliplr(mean_l-ci_margin_l)];

    hold on
    
    rci_plot = [mean_r+ci_margin_r, fliplr(mean_r-ci_margin_r)];
    plot(streamlines, mean_r, 'r', 'LineWidth', 1.2)
    fill(strm, rci_plot , 1, 'facecolor', 'red', 'edgecolor','none', 'facealpha', 0.2);
    plot(streamlines, mean_l, 'b', 'LineWidth', 1.2)
    fill(strm, lci_plot , 1, 'facecolor', 'blue', 'edgecolor', 'none', 'facealpha', 0.2);
    xlabel('Streamlines')
    ylabel('Mean FA')
    title("Subject " + string(i) + " FA Streamline Averages with 95% CI")
    legend('Right', '', 'Left', '', 'Location','north','Orientation','horizontal')
    
    saveas(gcf, string(subjects(i)) + '_FA_Comparison.png')
end
