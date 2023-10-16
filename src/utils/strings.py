"""
  @Time : 2022/9/16 15:30 
  @Author : Ziqi Wang
  @File : strings.py 
"""

def float2fixlenstr(v, l, scale=0):
    v = v / 10 ** scale
    tmp = str(v)
    n = tmp.find('.')
    # if n >= l:
    #     return 'invalid'
    fmt = f'%.0{max(1, l-n-1)}f'
    return fmt % v
    pass


if __name__ == '__main__':
    x = 2.3504365436
    print(float2fixlenstr(x, 5, 2))
    print(float2fixlenstr(x, 5, 1))
    print(float2fixlenstr(x, 5, 0))
    print(float2fixlenstr(x, 5, -1))
    print(float2fixlenstr(x, 5, -2))
    print(float2fixlenstr(x, 5, -3))
    print(float2fixlenstr(x, 5, -4))
    print('-' * 16)
    print(float2fixlenstr(-x, 5, 2))
    print(float2fixlenstr(-x, 5, 1))
    print(float2fixlenstr(-x, 5, 0))
    print(float2fixlenstr(-x, 5, -1))
    print(float2fixlenstr(-x, 5, -2))
    print(float2fixlenstr(-x, 5, -3))
