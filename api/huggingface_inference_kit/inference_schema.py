from typing import TypeAlias, TypeVar, Generic
from pydantic import BaseModel

InferenceSchema: TypeAlias = BaseModel

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')



from typing import Protocol, Annotated
from typing_extensions import Doc

class FileLike(Protocol):
    @property
    def filename(self) -> Annotated[str, Doc('Name of binary object')]:
        ...
    
    @property
    def file(self) -> Annotated[BinaryIO, Doc('Binary object')]:
        ...


Sequence, 
TypeVar,

T = TypeVar('T')
def func(l: List[T]) -> T:
    pass

class MyStuff(Generic[T])
    
Subclassing the Generic Type


alternative constructor
@classmethod
def from_string(cls, emp_str)
    first, last = empl_str.split('-')
    cls(first, last)

@staticmethod

__repr__(self)

__str__(self)

def __add__

def __len__

@overload (and true overloading can be done with metaclasses), Note typing class has @overload for static typing

