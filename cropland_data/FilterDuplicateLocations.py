# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   df = pd.read_csv('ground_truth_cropland.csv')
   SCALE = 100

   # Filter out till you get lat and lon utill 2 decimal places
   df['lat_idx'] = df.apply(lambda r : int(r.lat * SCALE), axis = 1)
   df['lon_idx'] = df.apply(lambda r : int(r.lon * SCALE), axis = 1)
   print(f'Number of rows before filtering: {df.shape}')


   # Filter out duplicates based on certain columns
   print('---- After Removing Duplicates ---')
   df = df.drop_duplicates(subset = ['year', 'month', 'lat_idx', 'lon_idx'])
   #print(df.head())
   print(f'Number of rows after filtering: {df.shape}')

   # Save filtered file to csv
   df = df.drop(['lat_idx', 'lon_idx'], axis = 1)
   print(df.head())
   df.to_csv(path_or_buf = f'out_{SCALE}.csv')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
