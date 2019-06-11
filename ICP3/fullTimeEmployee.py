#Importing Employee Class
from ICP3.employee import Employee
#Inheriting parent class Employee to child class FullTimeEmployee
class FullTimeEmployee(Employee):
    emp = Employee()
    print(emp.employeeDetails())
ftemp = FullTimeEmployee()
print(ftemp.employeeDetails())
print('Total Salary of all the employees',ftemp.totSal)
print('Average Salary of ',ftemp.noOfEmp,' Employees',ftemp.displayCount())


