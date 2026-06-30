#Task 1.2: Take a string input. Print its first character, last character, length, and the string in uppercase.

# Take string input
text = input("Enter a string: ")
# Print first character
print("First character:", text[0])
# Print last character
print("Last character:", text[-1])
# Print length of the string
print("Length:", len(text))
# Print the string in uppercase
print("Uppercase:", text.upper())

#Task 2.3: Keep asking the user for numbers (while loop). Stop when they enter 0. Print the sum of all numbers entered.
# Initialize the sum
total = 0
# Keep asking the user for numbers
while True:
    number = int(input("Enter a number (0 to stop): "))
    if number == 0:
        break
    total += number
# Print the sum
print("The sum of all numbers is:", total)

#Task 3.1: Create a list of 5 student names. Print them one by one using a loop.
# Create a list of 5 student names
students = ["Ali", "Ahmed", "Ayesha", "Fatima", "Sara"]

# Print each student name using a loop
for student in students:
    print(student)
    
