from types import new_class
from ObjectOrientedUtility import connectToDB as util


class testClass:
    'This is the first class that i create in Python'
    'some member varialbles'
    __testVar1 = 0

    def __init__(self, testVar1=0, name="") -> None:
        self.__testVar1 = testVar1
        self.name= name
        self.count = 0

    def displayTestVar1(self):
        testClass.__testVar1 += 1
        print("The test var 1 value is %d" %testClass.__testVar1)

    def displayCount(self):
        self.count += 1
        print("Count:", self.count)
    
    def displayName(self):
        print("Name:", self.name)   

class testClassSub(testClass):
    'showcase inheritance'

    def __init__(self, subName) -> None:
        self.subName = subName

    def subFunction(self):
        print("Sub Name:", self.subName)


testClassInstance = testClass(1190, "The game")
testClassInstance.displayTestVar1()
testClassInstance.displayTestVar1()
testClassInstance.displayName()
testClassInstance.displayCount()
testClassInstance.displayCount()

subClass = testClassSub("This is a sub name Clark")
subClass.subFunction()
subClass.displayTestVar1()

'Polymorphysm'
class proton:

    def __init__(self) -> None:
        print("init proton")
    
    def price(self):
        print("Proton is Cheap")

class bmw:

    def __init__(self) -> None:
        print('init bmw')
    
    def price(self):
        print("BMW is Expensive")


#common interface
def getPrice(car):
    car.price()

protonCar = proton()
bmwCar = bmw()

getPrice(protonCar)
getPrice(bmwCar)

'import other file class'
myUtil = util()
myUtil.testFunction()