def change(a, b):
    a.a = b


class faa:
    def __init__(self, b):
        self.a = b



a = faa(5)

b = 66
print(a.a)
change(a, b)

print(a.a)