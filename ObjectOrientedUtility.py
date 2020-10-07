'to demonstrate how other py file can refer to this file.'
class connectToDB:

    def __init__(self) -> None:
        print("init connectToDB")

    def testFunction(self):
        print("connectToDB called!!!")