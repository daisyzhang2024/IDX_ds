import pandas
import matplotlib.pyplot as plt

months = ['202512', '202601', '202602', '202603', '202604', '202605']

# Combine data from all months into a single DataFrame
dfs = []

for month in months:
    file_path = f'CRMLSSold{month}.csv'
    df = pandas.read_csv(file_path)
    dfs.append(df)

combined_df = pandas.concat(dfs, ignore_index=True) 
combined_df = combined_df[(combined_df['PropertyType'] == 'Residential') & (combined_df['PropertySubType'] == 'SingleFamilyResidence')]

for column in ['ClosePrice', 'LivingArea', 'BedroomsTotal', 'BathroomsTotalInteger', 'LotSizeSquareFeet']:
    # Remove outliers using the IQR method (use a copy to avoid filtering carrying over)
    col_data = combined_df[column].dropna()
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered = col_data[(col_data >= lower_bound) & (col_data <= upper_bound)]

    plt.figure()  # fresh figure each iteration
    filtered.hist(bins=50)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')
    plt.savefig(f'01_{column}_Distribution.png')
    plt.close()

    # Describe the filtered data
    description = filtered.describe()
    print(description)