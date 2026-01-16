import os

class robot:
    def __init__(self, name, battery):
        self.name = name
        self.battery = battery

    def work(self):
        print(f"{self.name} is working on laundry!")
        if self.battery > 20:
            self.battery -= 10

        # Reset battery when it reaches 10% or below
        if self.battery <= 10:
            print(f"Battery critically low at {self.battery}%! Resetting to 100%")
            self.battery = 100

    def save_battery(self, filename="robot_battery.txt"):
        with open(filename, "w") as f:
            f.write(str(self.battery))

    def load_battery(self, filename="robot_battery.txt"):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.battery = int(f.read())
        return self.battery


# battery usage
my_robot = robot("RoboboHelper", 100)
my_robot.load_battery()

print(f"Current battery: {my_robot.battery}%")
my_robot.work()
print(f"Life remaining: {my_robot.battery}%")

if my_robot.battery <= 20:
    print("Battery low! Please recharge.")

my_robot.save_battery()
