## ** `os` Module in Python**

The `os` module in Python provides a way to interact with the operating system, enabling developers to perform operations like file handling, directory management, and environment variable access.

### **Key Features of the `os` Module:**

- **File/Directory Management**: Create, delete, rename, or move files and directories.
- **Path Handling**: Interact with file paths for cross-platform compatibility.
- **Environment Variables**: Access and manipulate environment variables.
- **Process Management**: Manage and control system processes.

### **Commonly Used Functions of `os` Module:**

1. `os.getcwd()`: Returns the current working directory.
2. `os.chdir(path)`: Changes the current working directory.
3. `os.listdir(path)`: Lists files and directories in the specified path.
4. `os.mkdir(path)`: Creates a new directory.
5. `os.rmdir(path)`: Removes an empty directory.
6. `os.remove(path)`: Removes a file.
7. `os.rename(src, dst)`: Renames a file or directory.
8. `os.path.join(path, *paths)`: Joins multiple paths into one.
9. `os.path.exists(path)`: Checks if a path exists.
10. `os.environ`: Access environment variables.

### **Example Use Cases of the `os` Module**

#### **1. Get the Current Working Directory**

```python
import os

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)
```

#### **2. Create a New Directory**

```python
import os

new_dir = 'new_folder'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
    print(f"Directory '{new_dir}' created.")
else:
    print(f"Directory '{new_dir}' already exists.")
```

#### **3. List Files in a Directory**

```python
import os

path = '.'
files = os.listdir(path)
print(f"Files and Directories in '{path}':", files)
```

#### **4. Rename a File**

```python
import os

old_name = 'old_file.txt'
new_name = 'new_file.txt'

if os.path.exists(old_name):
    os.rename(old_name, new_name)
    print(f"Renamed '{old_name}' to '{new_name}'.")
else:
    print(f"File '{old_name}' not found.")
```

#### **5. Access Environment Variables**

```python
import os

user_name = os.environ.get('USER', 'Unknown')
print(f"Logged in as: {user_name}")
```

---

## ** `datetime` Module in Python**

The `datetime` module in Python allows for easy manipulation and formatting of dates and times. It includes classes for both date and time objects, making it useful for managing timestamps, intervals, and formatted date strings.

### **Key Features of the `datetime` Module:**

- **Date and Time Representation**: Work with current, past, or future dates and times.
- **Arithmetic on Dates**: Calculate differences between dates, add/subtract days, etc.
- **Formatting**: Convert dates into readable formats or custom formats.
- **Timezone Support**: Work with timezone-aware dates and times.

### **Commonly Used Classes and Functions in `datetime` Module:**

1. `datetime.datetime.now()`: Returns the current date and time.
2. `datetime.date.today()`: Returns the current date.
3. `datetime.timedelta`: Represents the difference between two dates or times.
4. `datetime.strptime()`: Converts a string to a `datetime` object using a format.
5. `datetime.strftime()`: Converts a `datetime` object to a string based on a format.
6. `datetime.date(year, month, day)`: Creates a date object.
7. `datetime.time(hour, minute, second)`: Creates a time object.

### **Example Use Cases of the `datetime` Module**

#### **1. Get Current Date and Time**

```python
from datetime import datetime

now = datetime.now()
print("Current Date and Time:", now)
```

#### **2. Get Only the Current Date**

```python
from datetime import date

today = date.today()
print("Today's Date:", today)
```

#### **3. Calculate Difference Between Two Dates**

```python
from datetime import datetime

date1 = datetime(2023, 5, 15)
date2 = datetime(2024, 10, 1)

difference = date2 - date1
print(f"Difference between {date1} and {date2}: {difference.days} days")
```

#### **4. Format a Date**

```python
from datetime import datetime

now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
print("Formatted Date and Time:", formatted_date)
```

#### **5. Parse a Date from String**

```python
from datetime import datetime

date_string = "2024-10-01 12:30:00"
date_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
print("Parsed Date Object:", date_object)
```

#### **6. Add or Subtract Time Using `timedelta`**

```python
from datetime import datetime, timedelta

now = datetime.now()
new_date = now + timedelta(days=10)
print("Date 10 Days From Now:", new_date)
```


