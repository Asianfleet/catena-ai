class a:
    pass

class b(a):
    pass

print(b.mro()[0]==b)