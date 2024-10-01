Pandas provides a powerful and easy-to-use interface for importing and exporting data in various formats like CSV, Excel, JSON, and SQL. Here's a comprehensive guide on how to handle data import and export using Pandas.

---

## **Data Import Using Pandas**

Pandas supports reading data from various file formats. Below are some common methods:

### **1. Reading CSV Files**

The `read_csv()` function is used to read a comma-separated values (CSV) file into a Pandas DataFrame.

#### **Example:**

```python
import pandas as pd

# Reading data from a CSV file
data = pd.read_csv('data.csv')
print(data.head())  # Displays the first 5 rows
```

You can specify additional parameters like delimiter, header row, etc.

#### **Reading a CSV with a Custom Delimiter:**

```python
data = pd.read_csv('data.csv', delimiter=';')  # Specify delimiter if not comma-separated
```

### **2. Reading Excel Files**

The `read_excel()` function reads data from an Excel file (.xls or .xlsx) into a DataFrame. You can specify the sheet name as well.

#### **Example:**

```python
# Reading data from an Excel file
data = pd.read_excel('data.xlsx', sheet_name='Sheet1')
print(data.head())
```

### **3. Reading JSON Files**

The `read_json()` function reads a JSON file or JSON string into a DataFrame.

#### **Example:**

```python
# Reading data from a JSON file
data = pd.read_json('data.json')
print(data.head())
```

### **4. Reading Data from SQL Databases**

Pandas can also read data directly from SQL databases using the `read_sql()` function. You need to establish a connection to the database using libraries like `sqlite3` or `SQLAlchemy`.

#### **Example (SQLite):**

```python
import sqlite3

# Connect to the database
conn = sqlite3.connect('database.db')

# Querying data from a table
data = pd.read_sql('SELECT * FROM table_name', conn)
print(data.head())

# Close the connection
conn.close()
```

### **5. Reading from HTML Tables**

Pandas can also scrape and read data from HTML tables using the `read_html()` function.

#### **Example:**

```python
# Reading data from HTML tables on a webpage
url = 'https://example.com/tables_page'
data = pd.read_html(url)
print(data[0].head())  # Prints the first table from the page
```

---

## **Data Export Using Pandas**

Pandas supports exporting DataFrames into various formats. Below are some common methods:

### **1. Exporting to CSV**

The `to_csv()` function is used to export a DataFrame to a CSV file.

#### **Example:**

```python
# Exporting DataFrame to a CSV file
data.to_csv('output.csv', index=False)  # Set index=False to not include the index column
```

You can specify additional parameters like custom delimiter, header options, etc.

#### **Export with Custom Delimiter:**

```python
data.to_csv('output.csv', sep=';', index=False)  # Custom delimiter
```

### **2. Exporting to Excel**

The `to_excel()` function exports the DataFrame to an Excel file. You can specify sheet names and more.

#### **Example:**

```python
# Exporting DataFrame to an Excel file
data.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
```

### **3. Exporting to JSON**

The `to_json()` function is used to export a DataFrame to a JSON file or string.

#### **Example:**

```python
# Exporting DataFrame to a JSON file
data.to_json('output.json', orient='records', lines=True)
```

The `orient` parameter defines the format of the JSON output. Common options are:

- `'records'`: List of dictionaries
- `'split'`: Separate arrays for index, columns, and data
- `'table'`: Nested JSON suitable for table representation

### **4. Exporting to SQL Database**

The `to_sql()` function is used to write records stored in a DataFrame to a SQL database.

#### **Example (SQLite):**

```python
import sqlite3

# Connect to the database
conn = sqlite3.connect('database.db')

# Export DataFrame to SQL table
data.to_sql('table_name', conn, if_exists='replace', index=False)

# Close the connection
conn.close()
```

- `if_exists='replace'`: This option replaces the table if it exists. Other options are `'append'` or `'fail'`.

### **5. Exporting to HTML**

The `to_html()` function exports a DataFrame to an HTML table.

#### **Example:**

```python
# Exporting DataFrame to an HTML file
data.to_html('output.html', index=False)
```

---

### **Summary of Common Methods for Import and Export:**

| **File Format** | **Import Function** | **Export Function** |
| --------------- | ------------------- | ------------------- |
| CSV             | `pd.read_csv()`     | `data.to_csv()`     |
| Excel           | `pd.read_excel()`   | `data.to_excel()`   |
| JSON            | `pd.read_json()`    | `data.to_json()`    |
| SQL             | `pd.read_sql()`     | `data.to_sql()`     |
| HTML            | `pd.read_html()`    | `data.to_html()`    |

---

These methods demonstrate how Pandas can handle various file formats, making it versatile for data manipulation and storage in data science projects.
