### Python `if` Statement and Its Variations: A Detailed Tutorial

The `if` statement in Python is a control flow tool that allows you to execute a block of code only if a specific condition is met. By combining `if` with `else` and `elif`, you can build more complex conditions for controlling the flow of your program. This tutorial will explore all possible variations of `if` statements, including using control statements like `pass`, `continue`, and `break`.

---

### Basic `if` Statement

The basic `if` statement evaluates a condition (expression), and if it is `True`, the block of code under the `if` statement is executed.

#### Syntax:

```python
if condition:
    # Block of code
```

#### Example:

```python
x = 10

if x > 5:
    print("x is greater than 5")

# Output:
# x is greater than 5
```

In this case, the condition `x > 5` is `True`, so the block inside the `if` statement is executed.

---

### `if-else` Statement

The `if-else` statement is used when you want to execute one block of code if the condition is `True` and a different block of code if the condition is `False`.

#### Syntax:

```python
if condition:
    # Block of code when condition is True
else:
    # Block of code when condition is False
```

#### Example:

```python
x = 3

if x > 5:
    print("x is greater than 5")
else:
    print("x is less than or equal to 5")

# Output:
# x is less than or equal to 5
```

Since `x = 3`, the condition `x > 5` is `False`, so the code under `else` is executed.

---

### `if-elif-else` (Multiple Conditions)

The `if-elif-else` statement is used to evaluate multiple conditions. The `elif` stands for "else if" and allows you to check additional conditions if the previous ones are `False`.

#### Syntax:

```python
if condition1:
    # Block of code when condition1 is True
elif condition2:
    # Block of code when condition1 is False and condition2 is True
else:
    # Block of code when all conditions are False
```

#### Example:

```python
x = 7

if x > 10:
    print("x is greater than 10")
elif x > 5:
    print("x is greater than 5 but less than or equal to 10")
else:
    print("x is 5 or less")

# Output:
# x is greater than 5 but less than or equal to 10
```

In this case, since `x = 7`, the condition `x > 5` is `True`, so the corresponding block is executed.

---

### Nested `if` Statements

You can nest `if` statements inside other `if` statements to evaluate conditions at multiple levels.

#### Example:

```python
x = 15

if x > 10:
    print("x is greater than 10")
    if x > 20:
        print("x is also greater than 20")
    else:
        print("x is 10 to 20")
else:
    print("x is 10 or less")

# Output:
# x is greater than 10
# x is 10 to 20
```

Here, after the first condition `x > 10` is evaluated as `True`, another `if` block checks whether `x > 20`.

---

### Using `pass` with `if`

The `pass` statement is used when a statement is syntactically required but you don’t want to execute any code yet. It’s often used as a placeholder.

#### Example:

```python
x = 5

if x > 5:
    pass  # Placeholder for future code
else:
    print("x is not greater than 5")

# Output:
# x is not greater than 5
```

The `pass` statement does nothing, allowing you to focus on other parts of the code or structure it for future changes.

---

### Using `continue` in `if` Statements

The `continue` statement inside a loop skips the rest of the code for the current iteration and moves on to the next iteration.

#### Example:

```python
for i in range(5):
    if i == 2:
        continue  # Skip printing when i is 2
    print(i)

# Output:
# 0
# 1
# 3
# 4
```

In this example, when `i == 2`, the `continue` statement is executed, and the loop moves to the next iteration without printing `2`.

---

### Using `break` in `if` Statements

The `break` statement is used to exit a loop prematurely when a certain condition is met.

#### Example:

```python
for i in range(5):
    if i == 3:
        break  # Stop the loop when i is 3
    print(i)

# Output:
# 0
# 1
# 2
```

Here, the loop terminates completely when `i == 3` because of the `break` statement.

---

### Ternary `if-else` (Conditional Expression)

Python supports a shorter form of `if-else` using the ternary operator, which allows you to write simple `if-else` conditions in a single line.

#### Syntax:

```python
value_if_true if condition else value_if_false
```

#### Example:

```python
x = 8
result = "Even" if x % 2 == 0 else "Odd"
print(result)

# Output:
# Even
```

In this example, the condition checks whether `x` is even or odd and assigns the result to `result` in one line.

---

### `if` Statement with Logical Operators (`and`, `or`, `not`)

You can combine multiple conditions using logical operators like `and`, `or`, and `not`.

#### Example 1: Using `and`

```python
x = 7

if x > 5 and x < 10:
    print("x is between 5 and 10")

# Output:
# x is between 5 and 10
```

#### Example 2: Using `or`

```python
x = 3

if x < 5 or x > 10:
    print("x is less than 5 or greater than 10")

# Output:
# x is less than 5 or greater than 10
```

#### Example 3: Using `not`

```python
x = False

if not x:
    print("x is False")

# Output:
# x is False
```

---

### Using `in` Keyword in `if` Statement

You can use the `in` keyword to check for membership in sequences like strings, lists, tuples, or dictionaries.

#### Example:

```python
vowels = ['a', 'e', 'i', 'o', 'u']
letter = 'e'

if letter in vowels:
    print(f"{letter} is a vowel")

# Output:
# e is a vowel
```

---

### Common Mistakes with `if` Statements

1. **Incorrect Indentation**: Python relies on indentation to define blocks of code. Make sure all code under an `if` statement is indented correctly.
   
   ```python
   x = 10
   if x > 5:
   print("This will raise an indentation error")  # This will cause an error
   ```

2. **Using `=` instead of `==`:** A common mistake is using a single equals sign (`=`) for comparison instead of the double equals (`==`).
   
   ```python
   x = 5
   if x = 5:  # This will raise a syntax error because `=` is an assignment operator
       print("x is 5")
   ```

---

### Exercises

#### Exercise 1: Check for Even or Odd

Write a program that takes an input number from the user and checks whether it is even or odd using an `if-else` statement.

```python
# Your code here
```

#### Exercise 2: Grade Classification

Write a program that takes a grade from the user and classifies it as follows:

- A grade above 90 is "Excellent"
- A grade between 70 and 90 is "Good"
- A grade between 50 and 70 is "Average"
- Below 50 is "Fail"

```python
# Your code here
```

#### Exercise 3: Multiples of 3 and 5

Write a program that iterates through numbers from 1 to 20. Use an `if` statement to print:

- `"Fizz"` for multiples of 3
- `"Buzz"` for multiples of 5
- `"FizzBuzz"` for multiples of both 3 and 5

```python
# Your code here
```

---

### Summary

- **`if` Statement**: Executes a block of code if the condition is `True`.
- **`if-else` Statement**: Executes one block of code if the condition is `True`, and another block if it is `False`.
- **`if-elif-else`**: Allows checking multiple conditions sequentially.
- **Ternary `if-else`**: A shorter way to write an `if-else` statement in a single line.
- **Control Statements**: `break`, `continue`, and `pass` can be used inside `if` statements to control the flow of loops.
