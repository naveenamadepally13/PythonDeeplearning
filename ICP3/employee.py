#creating a class
class Employee:
   #'Parent class'
   noOfEmp = 0
   totSal = 0
   # constructor for Employee class
   def __init__(self):
      self.name = input('Name :')
      self.family = input('Family :')
      self.department = input('Department :')
      self.salary = int(input('Salary :'))
      Employee.noOfEmp += 1
      Employee.totSal += self.salary

   def employeeDetails(self):
          employeeData = {}
          employeeData['Details of employee'] = Employee.noOfEmp
          employeeData['Name'] = self.name
          employeeData['Family'] = self.family
          employeeData['Department'] = self.department
          employeeData['Salary'] = self.salary
          return employeeData

   def displayCount(self):
        avg = Employee.totSal/Employee.noOfEmp
        return avg
