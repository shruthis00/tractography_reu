data_tot = readtable("compiled_data_fname"); % CHANGE ME to filename for compiled data 

n_str = data_tot.NumberOfStreamlines;
mean_fa = data_tot.MeanFA;
len_str = data_tot.MeanLengthOfStreamlines;

%L = 1, R = 2
side = 2;

if side == 1
    n_str = n_str(1:2:13);
    mean_fa = mean_fa(1:2:13);
    len_str = len_str(1:2:13);
elseif side == 2
    n_str = n_str(2:2:14);
    mean_fa = mean_fa(2:2:14);
    len_str = len_str(2:2:14);
end

figure(1)
C = [101,195,247]
C = C ./256;
scatter3(n_str, mean_fa, len_str, 'filled', 'MarkerFaceColor', C);

if side == 1
    title('Left Brain Scatter Plot')
elseif side == 2
    title('Right Brain Scatter Plot')
end
xlabel("Number of Streamlines")
ylabel("Mean FA Overall")
zlabel("Mean Streamline Length (mm)")

