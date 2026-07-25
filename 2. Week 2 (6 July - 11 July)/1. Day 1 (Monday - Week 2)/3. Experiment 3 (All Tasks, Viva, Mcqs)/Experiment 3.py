import numpy as np

# Create array of 10 numbers
arr = np.array([5, 12, 8, 20, 15, 7, 30, 18, 10, 25])

print("Array:", arr)
print("Sum:", np.sum(arr))
print("Mean:", np.mean(arr))
print("Max:", np.max(arr))
print("Min:", np.min(arr))
print("Standard Deviation:", np.std(arr))



import pandas as pd

# Create sample data
data = {
    'Name': ['Ali', 'Sara', 'Usman', 'Zara', 'Ahmed'],
    'Age': [25, 30, 28, 22, 35],
    'City': ['Haripur', 'Lahore', 'Karachi', 'Haripur', 'Islamabad'],
    'Salary': [50000, 65000, 58000, 47000, 72000]
}

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)

# Load CSV
df = pd.read_csv('data.csv')

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())




import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Filter: Salary > 55000
high_salary = df[df['Salary'] > 55000]
print("Employees with Salary > 55000:")
print(high_salary)

# Filter: City is Lahore
lahore_employees = df[df['City'] == 'Lahore']
print("\nEmployees from Lahore:")
print(lahore_employees)




import pandas as pd
import numpy as np

# Create data with missing values
data = {
    'Name': ['Ali', 'Sara', 'Usman', 'Zara', 'Ahmed'],
    'Age': [25, 30, np.nan, 22, 35],
    'Salary': [50000, 65000, 58000, np.nan, 72000]
}

df = pd.DataFrame(data)
print("Original:")
print(df)

# Check missing values
print("\nMissing values before:", df.isnull().sum().sum())

# Replace missing with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

print("\nAfter replacing:")
print(df)
print("Missing values after:", df.isnull().sum().sum())





import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Group by City and get mean Salary
city_mean = df.groupby('City')['Salary'].mean()
print("Mean Salary by City:")
print(city_mean)

# Group by City and get multiple stats
city_stats = df.groupby('City')['Salary'].agg(['mean', 'min', 'max'])
print("\nSalary Statistics by City:")
print(city_stats)




