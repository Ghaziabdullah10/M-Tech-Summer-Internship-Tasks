# Take input from the user
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Perform calculations
print("Sum =", num1 + num2)
print("Difference =", num1 - num2)
print("Product =", num1 * num2)
print("Quotient =", num1 / num2)

# Take temperature in Celsius
celsius = float(input("Enter temperature in Celsius: "))

# Convert to Fahrenheit
fahrenheit = (celsius * 9 / 5) + 32

# Display result
print("Temperature in Fahrenheit =", fahrenheit)

# Take user's name
name = input("Enter your name: ")

# Print greeting five times
for i in range(5):
    print("Hello", name)