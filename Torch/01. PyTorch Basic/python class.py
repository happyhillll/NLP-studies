# 함수와 클래스의 차이
# 1. 함수로 덧셈기 구현하기
result=0

def add(num):
    global result
    result += num
    return result

print(add(3))
print(add(4))

# 2. 함수로 두 개의 덧셈기 구현하기

result1=0
result2=0

def add1(num):
    global result1
    result1 += num
    return result1

def add2(num):
    global result2
    result2 += num
    return result2

print(add1(3))
print(add1(4))
print(add2(3))
print(add2(7))

# 3. 클래스로 덧셈기 구현하기

class Calculator:
    def __init__(self): # 생성자
        self.result=0
    def add(self,num):
        self.result += num
        return self.result

cal1=Calculator()
cal2=Calculator()

print(cal1.add(3))
print(cal1.add(4))
print(cal2.add(3))
print(cal2.add(7))