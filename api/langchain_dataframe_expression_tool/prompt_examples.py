examples = [
    {
        'input': 'Provide the number of rows and columns', 
        'output': 'df.shape'
    },
    {
        'input': 'Provide a summary of the data',
        'output': 'df.info()'
    },
    {
        'input': 'Show the last 5 rows in the data',
        'output': 'df.tail()'
    },
    {
        'input': 'Show the data in "column_a"',
        'output': 'df["column_a"]'
    },
    {
        'input': 'List the content of "column_a" and "column_b"',
        'output': 'df[["column_a", "column_b"]]'
    },
    {
        'input': 'List the column names',
        'output': 'df.columns'
    },
    {
        'input': 'List the content of the first row',
        'output': 'df.iloc[0]'
    },
    {
        'input': 'What is the content of the first and second row',
        'output': 'df.iloc[[0,1]]'
    },
    {
        'input': 'What is the content of the first and second row and the last column, given there are three total columns',
        'output': 'df.iloc[[0,1], 2]'
    },
    {
        'input': 'What is the content of the first and second row and the "column_a" column',
        'output': 'df.loc[[0,1], "column_a"]'
    },
    {
        'input': 'What is the content of the first and second row and the "column_a" and "column_b" columns',
        'output': 'df.loc[[0,1], "column_a", "column_b"]'
    },
    {
        'input': 'What are the values of the first three rows of the "column_a" column',
        'output': 'df.loc[0:2, "column_a"]'
    },
    {
        'input': 'What are the values of the first three rows of "column_a" to "column_d" columns',
        'output': 'df.loc[0:2, "column_a":"column_d"]'
    },
    {
        'input': 'Select rows from the range "value1" to "value10" in "column_a"',
        'output': 'df.set_index("column_a").loc["value1":"value10"]'
    },
    {
        'input': 'Select rows where "column_a" is equal to "value1"',
        'output': 'df[df["column_a"] == "value1"]'
    },
    {
        'input': 'Select values in column "column_a" where the "column_b" is equal to "value1"',
        'output': 'df.loc[df["column_b"] == "value1", "column_a"]'
    },
    {
        'input': 'Select rows where "column_a" column is equal to "value1" and "column_b" column is equal to "value2"',
        'output': 'df[(df["column_a"] == "value1") & (df["column_b"] == "value2")]'
    },
    {
        'input': 'Select rows where "column_a" column is equal to "value1" or "column_b" column is equal to "value2"',
        'output': 'df[(df["column_a"] == "value1") | (df["column_b"] == "value2")]'
    },
    {
        'input': 'Select rows where "column_a" column is not equal to "value1" or "column_b" column is not equal to "value2"',
        'output': 'df[~((df["column_a"] == "value1") | (df["column_b"] == "value2"))]'
    },
    {
        'input': 'Select "column_a" column, "column_b" column, and "column_c" column where "column_c" column is greater than 70000',
        'output': 'df.loc[df["column_c"] > 70000, "column_a":"column_c"]'
    },
    {
        'input': 'Select "column_a" column where "value1", "value2", "value3" are in "column_a"',
        'output': 'df.loc[df["column_a"].isin(["value1", "value2", "value3"]), "column_a"]'
    },
    {
        'input': 'Select "column_a" column where "value1", "value2", "value3" are not in "column_a"',
        'output': 'df.loc[~df["column_a"].isin(["value1", "value2", "value3"]), "column_a"]'
    },
    {
        'input': 'Select "column_a" where "column_a" contains text "value"',
        'output': 'df.loc[df["column_a"].str.contains("value"), "column_a"]'
    },
    {
        'input': 'Sort the data by column "column_a" in ascending order',
        'output': 'df.sort_values(by="column_a")'
    },
    {
        'input': 'Sort the data by column "column_a" in descending order',
        'output': 'df.sort_values(by="column_a", ascending=False)'        
    },
    {
        'input': 'Sort the data by column "column_a" and column "column_b" in descending order',
        'output': 'df.sort_values(by=["column_a", "column_b"], ascending=False)'
    },
    {
        'input': 'Sort the data by column "column_a" and column "column_b" with "column_a" in descending order and "column_b" in ascending order',
        'output': 'df.sort_values(by=["column_a", "column_b"], ascending=[False, True])'
    },
    {
        'input': 'Sort the data by column "column_a" and column "column_b" with "column_a" in descending order and "column_b" in ascending order and list the first 50 results',
        'output': 'df.sort_values(by=["column_a", "column_b"], ascending=[False, True]).head(50)'
    },
    {
        'input': 'What are the 10 largest values of column "column_a"',
        'output': 'df["column_a"].nlargest(10)'
    },
    {
        'input': 'Show all of the data of the 10 largest values of column "column_a"',
        'output': 'df.nlargest(10, "column_a")'
    },
    {
        'input': 'Show all of the data of the 10 smallest values of column "column_a"',
        'output': 'df.nsmallest(10, "column_a")'
    },
    {
        'input': 'What is the median value of column "column_a"',
        'output': 'df["column_a"].median()'
    },
    {
        'input': 'What are the median values of our data',
        'output': 'df.median()'
    },
    {
        'input': 'Provide some statistics of the data',
        'output': 'df.describe()'
    },
    {
        'input': 'Count the occurrences of unique values in column "column_a"',
        'output': 'df["column_a"].value_counts()'
    },
    {
        'input': 'Break down the count the occurrences of unique values in column "column_a" by percentage',
        'output': 'df["column_a"].value_counts(normalize=True)'
    },
    {
        'input': 'What is the counts of different values in column "column_a" broken down or grouped by column "column_b"',
        'output': 'df.groupby(["column_b"])["column_a"].value_counts()'
    },
    {
        'input': 'What is the median value of column "column_a" broken down or grouped by column "column_b"',
        'output': 'df.groupby(["column_b"])["column_a"].median()'
    },
    {
        'input': 'What is the median value and the mean of column "column_a" broken down or grouped by column "column_b"',
        'output': 'df.groupby(["column_b])["column_a"].agg(["median", "mean"])'
    },
    {
        'input': 'How many contain "value1" for column "column_a" filtered by "value2" in "column_b"',
        'output': 'df[df["columnb"] == "value2"]["column_a"].str.contains("value1").sum()'
    },
    {
        'input': 'How many contain "value1" for column "column_a" broken down or grouped by "column_b"',
        'output': 'df.groupby(["column_a"])["column_b"].apply(lambda x: x.str.contains("value1").sum())'
    },
    {
        'input': 'List rows without any missing values',
        'output': 'df.dropna()'
    },
    {
        'input': 'List the data without columns with all missing values',
        'output': 'df.dropna(axis="columns", how="all")'
    },
    {
        'input': 'List the rows whose column "column_a" is not empty',
        'output': 'df.dropna(axis="index", how="any", subset=["column_a"])'
    },
    {
        'input': 'List the rows where at least "column_a" or "column_b" is not empty',
        'output': 'df.dropna(axis="index", how="all", subset=["column_a", "column_b"])'
    },
    {
        'input': 'List the data where empty values are filled with 0s',
        'output': 'df.fillna(0)'
    },
    {
        'input': 'List the data types of the columns',
        'output': 'df.dtypes'
    },
    {
        'input': 'Calculate the average of "column_a" ignoring empty values',
        'output': 'df["column_a"].replace(["", " ", "None"], np.nan).astype(float).dropna().mean()'
    },
    {
        'input': 'Calculate the average of "column_a" with empty values filled with 0s',
        'output': 'df["column_a"].replace(["", " ", "None"], np.nan).astype(float).fillna(0).mean()'
    },
    {
        'input': 'List the unique values of "column_a"',
        'output': 'df["column_a"].unique'
    },
    {
        'input': 'Show the date column "column_a" as day of the week',
        'output': 'df["column_a"].dt.day_name()'
    },
    {
        'input': 'Show the earliest date in the date column "column_a',
        'output': 'df["column_a"].min()'
    },
    {
        'input': 'Show the most recent date in the date column "column_a',
        'output': 'df["column_a"].max()'
    },
    {
        'input': 'Show data newer than 2024-01-01 in the date column "column_a"',
        'output': 'df[df["column_a"] >= 2024-01-01]'
    },
    {
        'input': 'Show data greater than 2024-01-01 and less than 2025-01-01 in the date column "column_a"',
        'output': 'df[(df["column_a"] >= 2024-01-01) & (df["column_a"] <= 2025-01-01)]'
    }
]