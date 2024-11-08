import numpy as np
import matplotlib.pyplot as plt
import random, copy, xlrd,time

station = 6
train = 11
popsize = 50
iteration=20

late_train=5-1
late_station=1
late_time_start=30
late_time_tiaozheng=10
# 从excel表格中读取数据
timetable = xlrd.open_workbook(r"正点列车时刻表.xlsx")
sheet1 = timetable.sheet_by_name('Sheet1')  # 获取其内容
normal_timetable=[]
for i in range(2, train + 2):
    normal_timetable.append(sheet1.col_values(i)[1:])   # 获取第i列内容
normal_timetable=np.array(normal_timetable).astype(int)

# 获取列车最小停站时间
stoptime = np.array([[3,0,0,2,0,2],[3,0,0,2,0,2],[3,0,0,2,0,2],[3,0,0,2,0,2],[4,0,0,4,0,3],[4,0,0,4,0,3],[4,0,0,4,0,3],[4,0,0,4,0,3],[3,0,0,3,0,2],[3,0,0,3,0,2],[3,0,0,3,0,2]])
# 获取列车最小运行时间
transporttime=np.array([[20,25,29,10,25],[20,25,29,10,25],[20,25,29,10,25],[20,25,29,10,25],[36,48,40,23,44],[36,48,40,23,44],[36,48,40,23,44],[36,48,40,23,44],[24,30,38,21,42],[24,30,38,21,42],[24,30,38,21,42]])
# 获取各列车权重
w=np.array([3,3,3,3,1,1,1,1,2,2,2])
# 获取各列车等级
l=np.array([1,1,1,1,3,3,3,3,2,2,2])
# 获取车站到发线数目
track=np.array([3,3,3,2,2,3])

# 目标函数，总加权到、发晚点时间最小
def fitness(late_timetable):
    latetime = 0
    for j in range(train):
        for i in range(2 * station):
            latetime += abs(w[j]*(late_timetable[j,i] - normal_timetable[j,i]))
    return latetime

#约束条件，在满足约束条件的基础上初始化、进化
def limit(late_timetable_all):
    for j in range(len(late_timetable_all)):
        for m in range(late_train, train):  # 如果发车时间晚点，则提速，运行时间按最小运行时间,如果提前到站，则到站时间按正点运行时间
            for n in range(1,  2*station-1, 2):
                if late_timetable_all[j][m, n] > normal_timetable[m, n]:
                    late_timetable_all[j][m, n + 1] =late_timetable_all[j][m, n]+transporttime[m,int((n+1)/2)-1]
                    if late_timetable_all[j][m, n + 1]<normal_timetable[m,n+1]:
                        late_timetable_all[j][m, n + 1] = normal_timetable[m, n + 1]
        ccc = []
        for r in range(train):  # 保证站点内最小停留时间约束
            cc = []
            for t in range(0, 2*station-1, 2):
                cc.append(late_timetable_all[j][r, t + 1] - late_timetable_all[j][r, t])
            ccc.append(cc)
        ccc = np.array(ccc)
        for m in range(late_train,train):
            for n in range(station):
                if (ccc[m, n] - stoptime[m, n]) < 0:
                    late_timetable_all[j][m, n * 2 + 1] = stoptime[m, n] + late_timetable_all[j][m, n * 2]
        for mm in range(late_train, train):  # 列车不能早发车
            for nn in range(1, 2*station-1, 2):
                while late_timetable_all[j][mm, nn] < normal_timetable[mm, nn]:
                    late_timetable_all[j][mm, nn] =late_timetable_all[j][mm, nn]+1
        for h in range(2*station):  # 保证车与车的运行间隔约束(行车途中不发生越行)
            for l in range(late_train,train-1):
                for k in range(l+1,train):
                    if late_timetable_all[j][l, h] < late_timetable_all[j][k, h]:
                        while late_timetable_all[j][k, h] - late_timetable_all[j][l, h] < 4:
                            late_timetable_all[j][k, h]+=1
                    else:
                        while late_timetable_all[j][l, h] - late_timetable_all[j][k, h] < 4:
                            late_timetable_all[j][l, h]+=1
        for i in range(1, station * 2-1, 2):  # 列车站间越行约束
            for h in range(late_train,train):
                for k in range(late_train,train):
                    if late_timetable_all[j][h, i] < late_timetable_all[j][k, i]:
                        if late_timetable_all[j][h, i + 1] >= late_timetable_all[j][k, i + 1]:
                            late_timetable_all[j][k, i + 1] = late_timetable_all[j][h, i + 1] + 4
        l = np.array([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2])
        for h in range(station):  # 站内低等级不能越行高等级
            for i in range(train - 1):
                for k in range(i, train):
                    if late_timetable_all[j][i, h * 2] < late_timetable_all[j][k, 2 * h]:
                        if late_timetable_all[j][i, h * 2 + 1] > late_timetable_all[j][k, 2 * h + 1]:
                            if l[i] < l[k]:
                                late_timetable_all[j][k, 2 * h + 1] = late_timetable_all[j][i, h * 2 + 1] + 7

        for h in range(late_train, train):  # 保证每列车各自的运行时间递增
            for l in range(2*station - 1):
                while late_timetable_all[j][h, l + 1] < late_timetable_all[j][h, l]:
                    late_timetable_all[j][h, l + 1] = late_timetable_all[j][h, l + 1] + 1
    adjust_timetable = late_timetable_all - normal_timetable
    return adjust_timetable    #减去正点时刻表的结果

#种群初始化，生成一组满足要求的解
def initialization():
    late_timetable_all= []
    for i in range(popsize):  # 随机生成列车调整时间
        late_timetable= []
        for i in range(train):
            X = np.random.randint(-late_time_start, late_time_start, size=2*station)
            late_timetable.append(X)
        late_timetable= np.array(late_timetable)  # 由已知晚点信息，调整列车时刻表
        late_timetable[0:late_train,:] = 0
        late_timetable[late_train, 2*late_station-2] = late_time_tiaozheng
        late_timetable_all.append(late_timetable)
    late_timetable_all = np.array(late_timetable_all)
    late_timetable_all = late_timetable_all + normal_timetable
    late_timetable_all = limit(late_timetable_all)

    late_timetable_all = late_timetable_all + normal_timetable
    best_fitness = []
    best_individual = []
    best_index = 0
    for i in range(1, popsize):
        if fitness(late_timetable_all[best_index]) > fitness(late_timetable_all[i]):
            best_index = i
    best_fitness.append(fitness(late_timetable_all[best_index]))
    best_individual.append(late_timetable_all[best_index])

    limit_list = np.zeros(popsize)
    return late_timetable_all,limit_list,best_fitness,best_individual      #减去正点时刻表的结果

def employed(adjust_timetable,limit_list):
    v=copy.deepcopy(adjust_timetable)
    adjust_timetable = adjust_timetable + normal_timetable
    best_individue_index=0
    for i in range(popsize):
        if fitness(adjust_timetable[i])<fitness(adjust_timetable[best_individue_index]):
            best_individue_index=i
    for i in range(popsize):
        h=random.choice([d for d in range(popsize) if d != i])
        hh = random.choice([d for d in range(popsize) if d != i])
        for k in range(late_train,train):
            for j in range(2*station):
            #j=random.randint(0,2*station-1)
                # v[i,k,j]=v[i,k,j]+int((random.uniform(-1,1))*(v[i,k,j]-v[h,k,j]))
                # v[i, k, j] = v[best_individue_index, k, j] + int((random.uniform(-1, 1)) * (v[i, k, j] - v[h, k, j]))
                v[i, k, j] = v[best_individue_index, k, j] + int((random.uniform(-1, 1)) * (v[hh, k, j] - v[h, k, j]))
                if v[i,k,j]<-late_time_start:
                    v[i, k, j]=-late_time_start#+random.random()*(v[best_individue_index, k, j]+late_time_start)
                elif v[i,k,j]>late_time_start:
                    v[i, k, j]=late_time_start#-random.random()*(late_time_start-v[best_individue_index, k, j])
        v[i,late_train,2*late_station-2]=late_time_tiaozheng
        # v[i,:,0:(2*late_station-2)]=0
        v[i, 0:late_train, :] = 0
    v=v+normal_timetable
    v = limit(v)
    v=v+normal_timetable
    for i in range(popsize):
        if fitness(v[i])<fitness(adjust_timetable[i]):
            adjust_timetable[i]=v[i]
            limit_list[i] = 0
        else:
            limit_list[i]+=1
    late_timetable_all = limit(adjust_timetable)
    return late_timetable_all,limit_list    #减去正点时刻表的结果

def onlooker(adjust_timetable,limit_list):
    v = copy.deepcopy(adjust_timetable)
    adjust_timetable = adjust_timetable + normal_timetable
    total_fitness = 0
    for i in range(popsize):
        total_fitness += 1 / fitness(adjust_timetable[i])
    probablityy = []
    for j in range(popsize):
        probablityy.append((1 / fitness(adjust_timetable[j])) / total_fitness)
    probablity = np.cumsum(probablityy)
    new_pop = []
    ms = []
    for l in range(popsize):
        ms.append(random.uniform(0, 1))
    for h in range(len(ms)):
        for j in range(len(probablity)):
            if ms[h] < probablity[j]:
                new_pop.append(v[j])
                break
    new_pop = np.array(new_pop)
    best_individue_index = 0
    for i in range(popsize):
        if fitness(new_pop[i]) < fitness(new_pop[best_individue_index]):
            best_individue_index = i
    for i in range(popsize):
        h = random.choice([d for d in range(popsize) if d != i])
        hh = random.choice([d for d in range(popsize) if d != i])
        for k in range(late_train,train):
            #j=random.randint(0,2*station-1)
            for j in range(0, 2 * station):
                # new_pop[i, k, j] = v[i, k, j] + int((random.uniform(-1, 1)) * (v[i, k, j] - v[h, k, j]))
                # new_pop[i, k, j] = v[best_individue_index, k, j] + int((random.uniform(-1, 1)) * (v[i, k, j] - v[h, k, j]))
                new_pop[i, k, j] = v[best_individue_index, k, j] + int((random.uniform(-1, 1)) * (v[hh, k, j] - v[h, k, j]))
                if new_pop[i, k, j] < -late_time_start:
                    new_pop[i, k, j] =  -late_time_start#+random.random()*(new_pop[best_individue_index, k, j]+late_time_start)
                elif new_pop[i, k, j] > late_time_start:
                    new_pop[i, k, j] = late_time_start#-random.random()*(late_time_start-new_pop[best_individue_index, k, j])
        new_pop[i, late_train, 2*late_station-2] =late_time_tiaozheng
        # new_pop[i, :, 0:(2 * late_station - 2)] = 0
        new_pop[i, 0:late_train, :] = 0
    new_pop = new_pop + normal_timetable
    new_pop = limit(new_pop)
    new_pop = new_pop + normal_timetable
    for i in range(popsize):
        if fitness(new_pop[i]) < fitness(adjust_timetable[i]):
            adjust_timetable[i] = new_pop[i]
            limit_list[i] = 0
        else:
            limit_list[i]+=1
    late_timetable_all = limit(adjust_timetable)
    return late_timetable_all,limit_list  # 减去正点时刻表的结果

def scout(adjust_timetable,limit_list):
    for i in range(popsize):
        if limit_list[i]>10:
            late_timetable = []
            for i in range(0, train):
                X = np.random.randint(-late_time_start, late_time_start, size=2*station)
                late_timetable.append(X)
            late_timetable = np.array(late_timetable)  # 由已知晚点信息，调整列车时刻表
            late_timetable[0:late_train,:] = 0
            late_timetable[late_train, 2*late_station-2] =late_time_tiaozheng
            adjust_timetable[i]=late_timetable
            limit_list[i]=0
    adjust_timetable+=normal_timetable
    adjust_timetable=limit(adjust_timetable)
    return adjust_timetable,limit_list

def new(adjust_timetable,limit_list):
    v=copy.deepcopy(adjust_timetable)
    adjust_timetable = adjust_timetable + normal_timetable
    best_individue_index=0
    for i in range(popsize):
        if fitness(adjust_timetable[i])<fitness(adjust_timetable[best_individue_index]):
            best_individue_index=i
    for i in range(popsize):
        r1=random.uniform(0,1)
        r2=random.uniform(0,1-r1)
        r3=1-r1-r2
        h=random.choice([d for d in range(popsize) if d != i])
        p=random.choice([d for d in range(popsize) if d != i])
        for k in range(late_train,train):
            for j in range(2*station):
            #j=random.randint(0,2*station-1)
                #v[i,k,j]=r1*v[i,k,j]+r2*v[best_individue_index,k,j]+r3*(v[p,k,j]-v[h,k,j])
                v[i, k, j] =int( r1 * v[i, k, j] + r2 * v[best_individue_index, k, j] + r3 * (v[p, k, j] - v[h, k, j]))
                #v[i, k, j] = v[best_individue_index, k, j] + int((random.uniform(-1, 1)) * (v[i, k, j] - v[h, k, j]))
                if v[i,k,j]<-late_time_start:
                    v[i, k, j]=-late_time_start
                elif v[i,k,j]>late_time_start:
                    v[i, k, j]=late_time_start
        v[i,late_train,2*late_station-2]=late_time_tiaozheng
        # v[i, :, 0:(2 * late_station - 2)] = 0
        v[i, 0:late_train, :] = 0
    v=v+normal_timetable
    v = limit(v)
    v=v+normal_timetable
    for i in range(popsize):
        if fitness(v[i])<fitness(adjust_timetable[i]):
            adjust_timetable[i]=v[i]
    late_timetable_all = limit(adjust_timetable)
    return late_timetable_all,limit_list    #减去正点时刻表的结果

def baoliuzuiyouzhi(adjust_timetable,best_fitness,best_individual):
    v = copy.deepcopy(adjust_timetable)
    v=v+normal_timetable
    best_index=0
    for i in range(1,popsize):
        if fitness(v[best_index])>fitness(v[i]):
            best_index=i
    if fitness(v[best_index])<best_fitness[-1]:
        best_fitness.append(fitness(v[best_index]))
        best_individual.append(v[best_index])
    else:
        best_fitness.append(best_fitness[-1])
        best_individual.append(best_individual[-1])
    return best_fitness,best_individual

start = time.time()
generation,limit_list,fit,individual = initialization()

aaa=[]
for i in range(iteration):
    aa = []
    generation,limit_list = employed(generation,limit_list)
    generation,limit_list = onlooker(generation,limit_list)
    generation,limit_list = scout(generation,limit_list)
    generation,limit_list = new(generation,limit_list)
    fit,individual=baoliuzuiyouzhi(generation,fit,individual)
    for j in range(popsize):
        #print(fitness(normal_timetable,generation[j] + normal_timetable))
        aa.append(fitness(generation[j] + normal_timetable))
    aaa.append(min(aa))
    print(aa)
    print(min(aa))

print(fit)
print(individual[-1])
print(individual[-1]-normal_timetable)

end = time.time()

X = np.arange(1, 21).astype(dtype=np.str)
plt.plot(X,fit[1:], 'r',label='Gbest')
plt.plot(X,aaa,'g',label='Pbest')


plt.title("Train operation adjustment 1",family = 'Times new roman')
plt.legend()
plt.show()

print(str(end-start))

a=individual[-1]

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
plt.figure(2)
color_value = {'1': 'midnightblue','2': 'mediumblue','3': 'c','4': 'orangered','5': 'm','6': 'fuchsia','7':'olive','8': 'midnightblue','9': 'mediumblue','10': 'c','11': 'orangered'}
ylist = [1,1,2,2,3,3,4,4,5,5,6,6]

for i in range(train):
    plt.plot(a[i],ylist)

plt.title("Train operation adjustment 1",family = 'Times new roman')
plt.xlabel('Time (min)', family = 'Times new roman')
plt.ylabel('Station', family = 'Times new roman')
plt.grid(True) #show the grid
plt.show()
