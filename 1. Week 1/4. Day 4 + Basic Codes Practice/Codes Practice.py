name = "Ahmed"
age = 25
print(f"Hello, {name}! You are {age} years old.")



a = 10
b = 3
print(f"Sum: {a + b}")
print(f"Difference: {a - b}")
print(f"Product: {a * b}")
print(f"Quotient: {a / b}")




students = ["Ayesha", "Bilal", "Fatima"]
for student in students:
    print(f"{student} is present today")




def check_pass_fail(marks):
    if marks >= 50:
        return "Pass"
    else:
        return "Fail"


print(f"Usman: {check_pass_fail(72)}")
print(f"Sana: {check_pass_fail(40)}")




count = 5
name = "Hamza"
while count > 0:
    print(f"{name}, {count} minutes left")
    count -= 1
print("Time's up!")