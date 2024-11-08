import numpy as np
import xlrd
import matplotlib.pyplot as plt

station=6
train=11
def readexcel(i):
    shikebiao = xlrd.open_workbook(r"正点列车时刻表.xlsx")
    sheet1 = shikebiao.sheet_by_name('Sheet1')  # 获取其内容
    cols = sheet1.col_values(i)  # 获取第i列内容
    return cols
a=[]
for i in range(2,train+2):
    b=readexcel(i)
    b=b[1:]
    a.append(b)
a=np.array(a)
a.astype(int)

train1=a[0]
train2=a[1]
train3=a[2]
train4=a[3]
train5=a[4]
train6=a[5]
train7=a[6]
train8=a[7]
train9=a[8]
train10=a[9]
train11=a[10]


color_value = {
    '1': 'midnightblue',
    '2': 'mediumblue',
    '3': 'c',
    '4': 'orangered',
    '5': 'm',
    '6': 'fuchsia',
    '7': 'olive',
    '8': 'midnightblue',
    '9': 'mediumblue',
    '10': 'c',
    '11': 'orangered',

}

ylist = [1,1,2,2,3,3,4,4,5,5,6,6]

for i in range(train):
    plt.plot(a[i],ylist)

plt.title("Punctual train diagram",family = 'Times new roman')
plt.xlabel('Time (min)', family = 'Times new roman')
plt.ylabel('Station', family = 'Times new roman')
plt.grid(True) #show the grid
plt.show()