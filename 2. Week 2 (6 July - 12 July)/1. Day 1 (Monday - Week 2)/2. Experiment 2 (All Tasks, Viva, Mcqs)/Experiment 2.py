# Task 1: Program to check if a number is even or odd

# Get input from user
number = int(input("Enter a number: "))

# Check if number is even or odd using modulo operator
if number % 2 == 0:
    print(f"{number} is an EVEN number.")
else:
    print(f"{number} is an ODD number.")

# Additional: Check multiple numbers
print("\n--- Check multiple numbers ---")
numbers = [4, 7, 10, 13, 0, -5]
for num in numbers:
    if num % 2 == 0:
        print(f"{num} is EVEN")
    else:
        print(f"{num} is ODD")

    # Task 2: Program to print multiplication table using for loop

    # Get the number from user
    num = int(input("Enter a number to see its multiplication table: "))

    # Print the table header
    print(f"\n{'=' * 40}")
    print(f"Multiplication Table of {num}")
    print('=' * 40)

    # For loop from 1 to 10
    for i in range(1, 11):
        print(f"{num} × {i:2d} = {num * i:3d}")

    # Extended version - user can choose range
    print("\n--- Extended Version ---")
    start = int(input("Enter start range: "))
    end = int(input("Enter end range: "))

    print(f"\nMultiplication Table of {num} (from {start} to {end})")
    print('-' * 35)
    for i in range(start, end + 1):
        print(f"{num} × {i:2d} = {num * i:4d}")

    # Task 3: Program that keeps asking for numbers until user enters 0

    print("Number Sum Calculator")
    print("Enter numbers to add. Enter 0 to stop and see the sum.")
    print("-" * 45)

    total_sum = 0
    number_count = 0
    number_list = []  # To store all entered numbers

    while True:
        # Get number from user
        num = float(input(f"Enter number {number_count + 1}: "))

        # Check if user wants to stop
        if num == 0:
            break

        # Add to total and count
        total_sum += num
        number_count += 1
        number_list.append(num)

    # Display results
    print("\n" + "=" * 45)
    print("RESULTS")
    print("=" * 45)

    if number_count > 0:
        print(f"Numbers entered: {number_list}")
        print(f"Total numbers: {number_count}")
        print(f"Sum: {total_sum}")
        print(f"Average: {total_sum / number_count:.2f}")
    else:
        print("No numbers were entered (excluding 0).")

    print(f"Sum of all numbers: {total_sum}")


# Task 4: Function that returns factorial of a number

def factorial(n):
    """
    Calculate factorial of a number.
    Factorial of n (n!) = n × (n-1) × (n-2) × ... × 1

    Parameters:
    n (int): Number to calculate factorial for

    Returns:
    int: Factorial of n, or error message for invalid input
    """

    # Check for invalid input
    if n < 0:
        return "Error: Factorial is not defined for negative numbers."

    # Base cases: 0! = 1 and 1! = 1
    if n == 0 or n == 1:
        return 1

    # Calculate factorial using loop
    result = 1
    for i in range(2, n + 1):s
        result *= i

    return result


# Alternative: Recursive version
def factorial_recursive(n):
    """Calculate factorial using recursion"""
    if n < 0:
        return "Error: Negative numbers not allowed"
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)


# Test the function
print("=" * 50)
print("FACTORIAL CALCULATOR")
print("=" * 50)

# Get user input
num = int(input("Enter a number: "))

# Calculate factorial using both methods
result_loop = factorial(num)
result_recursive = factorial_recursive(num)

# Display results
print(f"\n{num}! = {result_loop}")
print(f"Recursive result: {result_recursive}")

# Show step-by-step calculation
print("\n--- Step-by-step calculation ---")
if num >= 0 and num <= 10:
    calc = " × ".join(str(i) for i in range(2, num + 1))
    print(f"{num}! = {calc} = {result_loop}")

# Test multiple values
print("\n--- Test multiple values ---")
test_numbers = [0, 1, 5, 7, 10, -3]
for n in test_numbers:
    print(f"{n}! = {factorial(n)}")

# Task 5: Program to count frequency of each word in a sentence

import string


def count_word_frequency(sentence):
    """
    Count the frequency of each word in a sentence.

    Parameters:
    sentence (str): Input sentence

    Returns:
    dict: Dictionary with words as keys and frequencies as values
    """

    # Convert to lowercase and remove punctuation
    # Remove punctuation using string.punctuation
    translator = str.maketrans('', '', string.punctuation)
    cleaned_sentence = sentence.translate(translator)

    # Split into words
    words = cleaned_sentence.lower().split()

    # Count frequencies using dictionary
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    return word_count


# Alternative: Using get() method (more Pythonic)
def count_word_frequency_get(sentence):
    """Count word frequencies using dictionary get() method"""
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    cleaned = sentence.translate(translator).lower()
    words = cleaned.split()

    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1

    return word_count


# Main program
print("=" * 50)
print("WORD FREQUENCY COUNTER")
print("=" * 50)

# Get input from user
sentence = input("Enter a sentence: ")

# Count frequencies
frequencies = count_word_frequency(sentence)

# Display results
print("\n" + "=" * 50)
print("WORD FREQUENCIES")
print("=" * 50)

if frequencies:
    # Sort by frequency (highest first)
    sorted_words = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Word':<15} {'Frequency':<10} {'Visual'}")
    print("-" * 40)

    max_count = max(frequencies.values()) if frequencies else 0

    for word, count in sorted_words:
        # Create visual bar
        bar = "█" * count
        print(f"{word:<15} {count:<10} {bar}")

    print("\n" + "-" * 40)
    print(f"Total words: {sum(frequencies.values())}")
    print(f"Unique words: {len(frequencies)}")

    # Most and least frequent
    most_freq = max(frequencies, key=frequencies.get)
    least_freq = min(frequencies, key=frequencies.get)
    print(f"Most frequent word: '{most_freq}' ({frequencies[most_freq]} times)")
    print(f"Least frequent word: '{least_freq}' ({frequencies[least_freq]} times)")

else:
    print("No words found in the sentence.")