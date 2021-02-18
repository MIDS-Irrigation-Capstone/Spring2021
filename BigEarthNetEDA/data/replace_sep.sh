#replace all , sep in csv with ; as , is also used as element separator for the array elements in 2nd column

sed -i 's/, \[/; \[/g' train_labels.csv
sed -i 's/, \[/; \[/g' test_labels.csv
sed -i 's/, \[/; \[/g' val_labels.csv

# remove the single quotes for the labels
sed -i "s/; \[/;\[/g" train_labels.csv
sed -i "s/; \[/;\[/g" test_labels.csv
sed -i "s/; \[/;\[/g" val_labels.csv

# change the separator in big_earth_net_labels.txt to be '; ' instead of ' '
sed -i 's/ \[/;\[/g' big_earth_net_labels.txt
mv big_earth_net_labels.txt big_earth_net_labels.csv

sed -i "s/; \[/;\[/g" big_earth_net_labels.csv
