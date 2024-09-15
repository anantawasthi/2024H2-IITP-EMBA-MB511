# Working with Text in Python

---

### **Course Outline:**

1. **Introduction to Strings**
2. **Basic String Operations**
3. **String Slicing**
4. **String Methods**
5. **String Formatting**
6. **Escape Characters**
7. **Working with Files and Text**

---

### **1. Introduction to Strings**

In Python, strings are sequences of characters enclosed in single (`'`) or double (`"`) quotes.

**Example:**

```python
# Single and Double quotes
string1 = 'Hello, World!'
string2 = "Python is fun!"
print(string1, string2)
```

**Output:**

```
Hello, World! Python is fun!
```

---

### **2. Basic String Operations**

- **Concatenation:** Combining strings.
- **Repetition:** Repeating strings using `*`.
- **Length of a String:** Using `len()` function.

**Examples:**

```python
# Concatenation
greeting = "Hello" + " " + "World!"
print(greeting)

# Repetition
echo = "echo! " * 3
print(echo)

# Length of a string
length = len("Python")
print(length)
```

**Output:**

```
Hello World!
echo! echo! echo!
6
```

---

### **3. String Slicing**

You can extract parts of a string using slicing. The syntax is `string[start:end]`.

**Examples:**

```python
word = "DataScience"

# Slicing to get 'Data'
print(word[0:4])

# Slicing from start till index 4
print(word[:4])

# Slicing from index 4 to the end
print(word[4:])
```

**Output:**

```
Data
Data
Science
```

---

### **4. String Methods**

Python provides several built-in methods to manipulate strings.

- **`lower()` and `upper()`** – Converts string to lowercase or uppercase.
- **`strip()`** – Removes leading and trailing spaces.
- **`replace()`** – Replaces a substring with another.
- **`split()`** – Splits the string into a list.

**Examples:**

```python
text = "  Hello Python!  "

# Convert to lowercase
print(text.lower())

# Remove spaces
print(text.strip())

# Replace substring
print(text.replace("Python", "World"))

# Split string into a list
print(text.split(" "))
```

**Output:**

```
  hello python!  
Hello Python!
  Hello World!  
['', '', 'Hello', 'Python!', '', '']
```

---

### **5. String Formatting**

There are multiple ways to format strings in Python:

- **Using f-strings (Python 3.6+)**
- **Using `.format()` method**

**Examples:**

```python
name = "Alice"
age = 25

# f-string formatting
print(f"My name is {name} and I am {age} years old.")

# Using .format() method
print("My name is {} and I am {} years old.".format(name, age))
```

**Output:**

```
My name is Alice and I am 25 years old.
My name is Alice and I am 25 years old.
```

---

### **6. Escape Characters**

Escape characters allow special characters to be included in strings. Common escape characters are:

- **`\n`** – Newline
- **`\t`** – Tab
- **`\'`** – Single quote
- **`\"`** – Double quote

**Example:**

```python
# Newline and tab
text = "Hello\nPython\tWorld!"
print(text)
```

**Output:**

```
Hello
Python    World!
```

---

### **7. Working with Files and Text**

You can read from and write to files in Python. Here’s an example:

**Reading a File:**

```python
# Reading a file
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
```

**Writing to a File:**

```python
# Writing to a file
with open('output.txt', 'w') as file:
    file.write("Hello, World!")
```

---

### **Conclusion**

This quick course covers the basics of string operations in Python. By understanding these concepts, you'll be able to efficiently manipulate text data for various tasks such as parsing, formatting, and processing text.

Would you like to explore advanced string operations or practical examples next?
