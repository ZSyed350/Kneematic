import random

output_file = "visualize/generated_data.txt"
num_rows = 1000

with open(output_file, "w") as f:
    for _ in range(num_rows):
        first = random.choice([0, 1, 2, 3, 4, 5])
        middle = [round(random.uniform(0, 100), 2) for _ in range(6)]
        last = round(random.uniform(0, 90), 2)

        row = [first] + middle + [last]
        f.write(",".join(map(str, row)) + "\n")

print(f"{num_rows} rows written to {output_file}")