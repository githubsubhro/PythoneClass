class Person:
    def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname
        print(self)

    def printname(self):
        print(self, self.firstname, self.lastname)


class Student(Person):
  def __init__(self, fname, lname, year):
    super().__init__(fname, lname)
    self.graduationyear = year

  def welcome(self):
    print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)


# Use the Person class to create an object, and then execute the printname method:

x = Student("Subhro", "Doe", "1990")
print(x)
x.printname()
x.welcome()
