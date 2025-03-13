examples = [
    {"input": "Provide the number of rows and columns.", "output": "df.shape"},
    {"input": "Provide a summary of the data.", "output": "df.info()"},
    {"input": "Show the last 5 rows in the data.", "output": "df.tail()"},
    {"input": "List the column names.", "output": "df.columns"},
    {"input": "List the content of the first row.", "output": "df.iloc[0]"},
    {
        "input": "What is the content of the first and second row?",
        "output": "df.iloc[[0,1]]",
    },
    {"input": 'Show the data in "{col_name}".', "output": "df[{col_name!r}]"},
    {
        "input": 'What are the first 3 rows of the "{col_name}" column?',
        "output": "df.loc[0:2, {col_name!r}]",
    },
    {
        "input": 'List the unique values in "{col_name}".',
        "output": "df[{col_name!r}].unique()",
    },
    {
        "input": 'Select rows where "{col_name}" is equal to "value1".',
        "output": 'df[df[{col_name!r}] == "value1"]',
    },
    {
        "input": 'List the content of "{col1}" and "{col2}".',
        "output": "df[[{col1!r}, {col2!r}]]",
    },
    {
        "input": 'What is the content of rows 0 and 1 for the "{col_name}" column?',
        "output": "df.loc[[0, 1], {col_name!r}]",
    },
    {
        "input": "What is the content of the first and second row and the last column, given there are three total columns",
        "output": "df.iloc[[0,1], 2]",
    },
    {
        "input": 'What is the content of rows {row1} and {row2} for the "{col_a}" and "{col_b}" columns?',
        "output": "df.loc[[{row1}, {row2}], [{col_a!r}, {col_b!r}]]",
    },
    {
        "input": 'What are the values of the first three rows of the "{col_name}" column?',
        "output": "df.loc[0:2, {col_name!r}]",
    },
    {
        "input": 'What are the values of the first three rows from "{start_col}" to "{end_col}" columns?',
        "output": "df.loc[0:2, {start_col!r}:{end_col!r}]",
    },
    {
        "input": 'Sort the data by "{col1}" in ascending order.',
        "output": "df.sort_values(by={col1!r})",
    },
    {
        "input": 'Sort the data by "{col1}" in descending order.',
        "output": "df.sort_values(by={col1!r}, ascending=False)",
    },
    {
        "input": 'Sort the data by "{col1}" and "{col2}" in ascending order.',
        "output": "df.sort_values(by=[{col1!r}, {col2!r}])",
    },
    {
        "input": 'Sort the data by "{col1}" and "{col2}" in descending order.',
        "output": "df.sort_values(by=[{col1!r}, {col2!r}], ascending=False)",
    },
    {
        "input": 'Sort the data by "{col1}" and "{col2}" with "{col1}" in descending order and "{col2}" in ascending order.',
        "output": "df.sort_values(by=[{col1!r}, {col2!r}], ascending=[False, True])",
    },
    {
        "input": 'Sort the data by "{col1}" in descending order and "{col2}" in ascending order, then list the first {n} results.',
        "output": "df.sort_values(by=[{col1!r}, {col2!r}], ascending=[False, True]).head({n})",
    },
    {
        "input": 'Select rows from the range "{value1}" to "{value2}" in "{col_name}".',
        "output": "df.set_index({col_name!r}).loc[{value1!r}:{value2!r}]",
    },
    {
        "input": 'Select rows where "{col1}" equals "{value1}" and "{col2}" equals"{value2}".',
        "output": "df[(df[{col1!r}] == {value1!r}) & (df[{col2!r}] == {value2!r})]",
    },
    {
        "input": 'Select rows where "{col_name}" is equal to "{value}".',
        "output": "df[df[{col_name!r}] == {value!r}]",
    },
    {
        "input": 'Select values in column "{col1}" where "{col2}" is equal to "{value}".',
        "output": "df.loc[df[{col2!r}] == {value!r}, {col1!r}]",
    },
    {
        "input": 'Select rows where "{col1}" is equal to "{value1}" AND "{col2}" is equal to "{value2}".',
        "output": "df[(df[{col1!r}] == {value1!r}) & (df[{col2!r}] == {value2!r})]",
    },
    {
        "input": 'Select rows where "{col1}" is equal to "{value1}" OR "{col2}" is equal to "{value2}".',
        "output": "df[(df[{col1!r}] == {value1!r}) | (df[{col2!r}] == {value2!r})]",
    },
    {
        "input": 'Select rows where "{col1}" is NOT equal to "{value1}" OR "{col2}" is NOT equal to "{value2}".',
        "output": "df[~((df[{col1!r}] == {value1!r}) | (df[{col2!r}] == {value2!r}))]",
    },
    {
        "input": 'Select "{col1}", "{col2}", and "{col3}" columns where "{col3}" is greater than {threshold}.',
        "output": "df.loc[df[{col3!r}] > {threshold}, {col1!r}:{col3!r}]",
    },
    {
        "input": 'Select "{col1}" where "{col1}" is in the list "{value_list}".',
        "output": "df.loc[df[{col1!r}].isin({value_list!r}), {col1!r}]",
    },
    {
        "input": 'Select "{col1}" where "{col1}" is NOT in the list "{value_list}".',
        "output": "df.loc[~df[{col1!r}].isin({value_list!r}), {col1!r}]",
    },
    {
        "input": 'Select "{col1}" rows where "{col1}" contains text "{value}".',
        "output": "df.loc[df[{col1!r}].str.contains({value!r}), {col1!r}]",
    },
    {
        "input": 'What are the {n} largest values of "{col_name}"?',
        "output": "df[{col_name!r}].nlargest({n})",
    },
    {
        "input": 'Show all of the data of the {n} largest values of "{col_name}".',
        "output": "df.nlargest({n}, {col_name!r})",
    },
    {
        "input": 'Show all of the data of the {n} smallest values of "{col_name}".',
        "output": "df.nsmallest({n}, {col_name!r})",
    },
    {
        "input": 'What is the median value of "{col_name}"?',
        "output": "df[{col_name!r}].median()",
    },
    {"input": "What are the median values of the data?", "output": "df.median()"},
    {"input": "Describe the data.", "output": "df.describe()"},
    {
        "input": 'Count the occurrences of unique values in "{col_name}".',
        "output": "df[{col_name!r}].value_counts()",
    },
    {
        "input": 'Break down the count of unique values in "{col_name}" by percentage.',
        "output": "df[{col_name!r}].value_counts(normalize=True)",
    },
    {
        "input": 'What is the count of different values in "{col1}" broken down or grouped by "{col2}"?',
        "output": "df.groupby({col2!r})[{col1!r}].value_counts()",
    },
    {
        "input": 'What is the median value of "{col1}" broken down or grouped by "{col2}"?',
        "output": "df.groupby({col2!r})[{col1!r}].median()",
    },
    {
        "input": 'What are the median and mean of "{col1}" broken down or grouped by "{col2}"?',
        "output": 'df.groupby({col2!r})[{col1!r}].agg(["median", "mean"])',
    },
    {
        "input": 'How many entries in column "{col1}" contain "{value}", when column "{col2}" is equal to "{value2}"?',
        "output": "df[df[{col2!r}] == {value2!r}][{col1!r}].str.contains({value!r}).sum()",
    },
    {
        "input": 'How many rows in "{col2}" contain "{value}", broken down or grouped by "{col1}"?',
        "output": "df.groupby({col1!r})[{col2!r}].apply(lambda x: x.str.contains({value!r}).sum())",
    },
    {
        "input": "Remove columns that are entirely empty and display the remaining data.",
        "output": 'df.dropna(axis="columns", how="all")',
    },
    {
        "input": 'Drop rows with any missing values in "{col_name}".',
        "output": "df.dropna(axis='index', how='any', subset=[{col_name!r}])",
    },
    {
        "input": 'List the rows where at least "{col1}" or "{col2}" is not empty.',
        "output": "df.dropna(axis='index', how='all', subset=[{col1!r}, {col2!r}])",
    },
    {
        "input": "List the data where empty values are filled with 0s",
        "output": "df.fillna(0)",
    },
    {"input": "List the data types of the columns", "output": "df.dtypes"},
    {
        "input": 'Calculate the average of "{col_name}" ignoring empty values.',
        "output": "df[{col_name!r}].replace(['', ' ', 'None'], np.nan).astype(float).dropna().mean()",
    },
    {
        "input": 'Calculate the average of "{col_name}" with empty values filled with 0s.',
        "output": "df[{col_name!r}].replace(['', ' ', 'None'], np.nan).astype(float).fillna(0).mean()",
    },
    {
        "input": 'Show the day of the week for each date in "{col_name}".',
        "output": "df[{col_name!r}].dt.day_name()",
    },
    {
        "input": 'What\'s the earliest date listed in "{col_name}"?',
        "output": "df[{col_name!r}].min()",
    },
    {
        "input": 'Find the most recent date in "{col_name}".',
        "output": "df[{col_name!r}].max()",
    },
    {
        "input": 'List all rows where "{col_name}" is after "{start_date}".',
        "output": "df[df[{col_name!r}] >= {start_date!r}]",
    },
    {
        "input": 'Show data greater than "{start_date}" and less than "{end_date}" in the column "{col_name}".',
        "output": "df[(df[{col_name!r}] >= {start_date!r}) & (df[{col_name!r}] <= {end_date!r})]",
    },
]
