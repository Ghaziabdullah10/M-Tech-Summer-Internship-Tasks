
tasks = []

def show_menu():
    print("\n========== TO-DO LIST MENU ==========")
    print("1. View all tasks")
    print("2. Add a new task")
    print("3. Mark a task as done")
    print("4. Delete a task")
    print("5. Exit")
    print("=======================================")

def view_tasks():

    if len(tasks) == 0:
        print("\nYour to-do list is empty!")
        return

    print("\n----- YOUR TASKS -----")

    for i, t in enumerate(tasks, start=1):

        if t["done"] == True:
            status = "[DONE]"
        else:
            status = "[ ]"
        print(i, status, t["task"])

def add_task():
    new_task = input("\nEnter the task you want to add: ")

    task_dict = {"task": new_task, "done": False}

    tasks.append(task_dict)

    print("Task added successfully!")

def mark_done():
    view_tasks()

    if len(tasks) == 0:
        return

    choice = int(input("\nEnter the task number to mark as done: "))
    index = choice - 1


    if index >= 0 and index < len(tasks):
        tasks[index]["done"] = True
        print("Task marked as done!")
    else:
        print("Invalid task number.")

def delete_task():
    view_tasks()

    if len(tasks) == 0:
        return

    choice = int(input("\nEnter the task number to delete: "))
    index = choice - 1

    if index >= 0 and index < len(tasks):
        removed = tasks.pop(index)
        print("Deleted task:", removed["task"])
    else:
        print("Invalid task number.")


while True:
    show_menu()
    user_choice = input("Enter your choice (1-5): ")

    if user_choice == "1":
        view_tasks()
    elif user_choice == "2":
        add_task()
    elif user_choice == "3":
        mark_done()
    elif user_choice == "4":
        delete_task()
    elif user_choice == "5":
        print("\nGoodbye! Thanks for using the To-Do List app.")
        break
    else:
        print("\nInvalid choice. Please enter a number between 1 and 5.")