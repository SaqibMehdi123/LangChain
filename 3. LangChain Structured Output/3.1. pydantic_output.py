# Pydantic is a data validation and data parsing library in Python. It uses Python type annotations to define the structure of data and provides runtime validation and parsing. It ensures that the data you work with is type-safe, correct and structured.
# Pydantic also validates emails, URLs, dates, and more complex data types.

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    # name: str
    name: str = 'Saqib' # setting default value
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, description="CGPA must be between 0 and 10")

new_student = {'age': 20, 'email': 'saqib@gmail.com', 'cgpa': 5}
# new_student = {'age': '20'} # pydantic converts str to int on its own (Coercing)
# new_student = {'age': 20}
# new_student = {}
# new_student = {'name': 'Saqib'}
# new_student = {'name': 20} # should raise an error

student = Student(**new_student)

print(student)
print(type(student))

student_dict = student.model_dump() # model_dump() converts pydantic model to dictionary
print(student_dict)
print(type(student_dict))

student_json = student.model_dump_json() # model_dump_json() converts pydantic model to JSON string
print(student_json)
print(type(student_json))