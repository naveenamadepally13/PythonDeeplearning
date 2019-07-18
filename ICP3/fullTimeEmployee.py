#Importing Employee Class
from ICP3.employee import Employee
#Inheriting parent class Employee to child class FullTimeEmployee
class FullTimeEmployee(Employee):
    emp = Employee()
    emp.__init__()
    def __init__(self):
        self.name = input('Full Time emp Name :')
        self.family = input('Full Time emp Family :')
        self.department = input('Full Time emp Department :')
        self.salary = int(input('Full Time emp Salary :'))
        Employee.noOfEmp += 1
        Employee.totSal += self.salary
        fullEmpDetails = Employee.employeeDetails(self)
        return fullEmpDetails

    print(emp.employeeDetails())
    FullTimeEmployee.ftemp = FullTimeEmployee()
    print(ftemp.employeeDetails())
# print(ftemp.fullTimeEmployeeDetails())
# #print(ftemp.fullTimeEmployeeDetails())
# print('Total Salary of all the employees',ftemp.totSal)
# print('Average Salary of ',ftemp.noOfEmp,' Employees',ftemp.displayCount())
#     emp = Employee()
#     print(emp.employeeDetails())
#     FullTimeEmployee.ftemp = FullTimeEmployee()
#     print(ftemp.employeeDetails())
# print(ftemp.fullTimeEmployeeDetails())
# #print(ftemp.fullTimeEmployeeDetails())
# print('Total Salary of all the employees',ftemp.totSal)
# print('Average Salary of ',ftemp.noOfEmp,' Employees',ftemp.displayCount())


