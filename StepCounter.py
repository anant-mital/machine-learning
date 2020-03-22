
steps = []
print("Enter the amount of steps/day, one day per line.\nEnd by entering an empty row.",end="")

counter = 0
steps.append(input())
while True:
    steps.append(input())
    # Checking if last step is empty
    if steps[counter + 1] == "":
        break
    else:
        counter = counter + 1
print(steps)
    
    
    
