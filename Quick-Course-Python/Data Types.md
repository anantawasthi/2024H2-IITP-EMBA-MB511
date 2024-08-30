#### **Basic Data Types:**

- **Integers (`int`):**
  - Whole numbers, positive or negative, without decimals.
  - Example: `x = 10`, `y = -5`
- **Floating-Point Numbers (`float`):**
  - Numbers that contain a decimal point.
  - Example: `pi = 3.14`, `temperature = -5.6`
- **Strings (`str`):**
  - A sequence of characters enclosed in quotes, used for text.
  - Example: `name = "Alice"`, `greeting = 'Hello'`
- **Booleans (`bool`):**
  - Represents one of two values: `True` or `False`.
  - Example: `is_valid = True`, `has_access = False`



**Conversion of Basic Data Types in Python**

- **Implicit Type Conversion:**
  
  - Python automatically converts one data type to another when necessary.
  - Example:
    
    ```python
    x = 5      # int
    y = 3.2    # float
    result = x + y  # result will be 8.2 (float)
    ```

- **Explicit Type Conversion:**
  
  - **int()**: Converts a value to an integer.
    - Example: 
      
      ```python
      x = int(3.8)  # x becomes 3
      y = int("10")  # y becomes 10
      ```
  - **float()**: Converts a value to a floating-point number.
    - Example:
      
      ```python
      x = float(5)  # x becomes 5.0
      y = float("7.2")  # y becomes 7.2
      ```
  - **str()**: Converts a value to a string.
    - Example:
      
      ```python
      x = str(100)  # x becomes "100"
      y = str(3.14)  # y becomes "3.14"
      ```
  - **bool()**: Converts a value to a boolean (`True` or `False`).
    - Example:
      
      ```python
      x = bool(1)  # x becomes True
      y = bool(0)  # y becomes False
      z = bool("")  # z becomes False
      ```

- **Examples of Type Conversion:**
  
  - **From String to Integer:**
    
    ```python
    num_str = "123"
    num_int = int(num_str)  # num_int becomes 123
    ```
  - **From Integer to String:**
    
    ```python
    num = 456
    num_str = str(num)  # num_str becomes "456"
    ```
  - **From Integer to Float:**
    
    ```python
    x = 7
    y = float(x)  # y becomes 7.0
    ```
  - **From Float to Integer:**
    
    ```python
    z = 3.99
    w = int(z)  # w becomes 3 (truncates decimal part)
    ```


