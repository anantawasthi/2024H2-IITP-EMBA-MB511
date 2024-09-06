In Python, **mutability** refers to whether an object’s state (i.e., its data or content) can be changed after it is created. Objects that allow changes are called **mutable**, while those that don’t are termed **immutable**. Let's dive into what this means for different types of Python objects.

### Mutable Objects

Mutable objects are those that can be modified after they are created. For example, elements can be added, removed, or changed without having to create a new object. This feature allows efficient in-place modifications.

#### Common Mutable Types:

1. **Lists**:
   
   - You can modify a list (add, remove, change elements) after it's created.
   - Example:
     
     ```python
     my_list = [1, 2, 3]
     my_list[0] = 10  # Modify the first element
     print(my_list)  # Output: [10, 2, 3]
     ```

2. **Dictionaries**:
   
   - Keys and values can be added, removed, or updated.
   - Example:
     
     ```python
     my_dict = {"name": "Alice", "age": 25}
     my_dict["age"] = 26  # Modify the value for the key 'age'
     print(my_dict)  # Output: {'name': 'Alice', 'age': 26}
     ```

3. **Sets**:
   
   - Sets can have elements added or removed.
   - Example:
     
     ```python
     my_set = {1, 2, 3}
     my_set.add(4)  # Adding an element
     print(my_set)  # Output: {1, 2, 3, 4}
     ```

4. **Bytearray**:
   
   - Mutable sequence of bytes.
   - Example:
     
     ```python
     byte_arr = bytearray([1, 2, 3])
     byte_arr[1] = 100  # Modify the second byte
     print(byte_arr)  # Output: bytearray(b'\x01d\x03')
     ```

### Immutable Objects

Immutable objects are those whose state cannot be changed after they are created. Any modification to these objects results in the creation of a new object rather than altering the original object.

#### Common Immutable Types:

1. **Integers**:
   
   - Once an integer object is created, its value cannot be changed.
   - Example:
     
     ```python
     x = 5
     x = 6  # A new integer object is created, the old one (5) is discarded.
     ```

2. **Strings**:
   
   - Strings cannot be altered in place. Any change will result in a new string being created.
   - Example:
     
     ```python
     my_str = "hello"
     my_str = my_str + " world"  # A new string "hello world" is created
     print(my_str)  # Output: 'hello world'
     ```

3. **Tuples**:
   
   - Tuples are immutable; once defined, the elements within a tuple cannot be modified.
   - Example:
     
     ```python
     my_tuple = (1, 2, 3)
     # my_tuple[0] = 10  # This will raise an error since tuples are immutable
     ```

4. **Frozensets**:
   
   - A frozenset is an immutable version of a set.
   - Example:
     
     ```python
     my_frozenset = frozenset([1, 2, 3])
     # my_frozenset.add(4)  # This will raise an error because frozensets are immutable
     ```

5. **Booleans**:
   
   - Like integers, booleans are immutable.
   - Example:
     
     ```python
     x = True
     x = False  # A new boolean object is created
     ```

6. **Bytes**:
   
   - A sequence of immutable bytes.
   - Example:
     
     ```python
     my_bytes = b"hello"
     # my_bytes[0] = 102  # This will raise an error since bytes are immutable
     ```

### Summary of Mutable vs. Immutable

- **Mutable Objects**: Lists, dictionaries, sets, bytearrays.
  
  - Their contents can be changed in place without creating a new object.
  - Useful when you want to make changes to the same object over time.

- **Immutable Objects**: Integers, strings, tuples, frozensets, bytes, booleans.
  
  - Once created, their contents cannot be modified.
  - If you attempt to modify them, a new object is created, and the original remains unchanged.


