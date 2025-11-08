import numpy as np
class Simplex():#默认为minmize
    def __init__(self,n_params,C,A,b,equal):
        self.n_params = n_params #初始的变量数
        self.C = C
        self.A = A
        self.b = b
        self.equal = equal
        self.object = C.copy()
        self.Cb = np.zeros((self.equal.shape[0],1))
        self.Cn = C.copy()
        self.M = 1e10
        self.base_index = np.zeros((self.equal.shape[0])) #基变量的坐标，是向量
        self.non_base_index = np.arange(self.n_params) #非基变量的初始坐标（后续会不断更新），是向量
        self.check = None #检验数
        
    def Generate_Simplex_Table(self):#构建初始的单纯形表
        for i in range(self.A.shape[0]):
            new_column = np.zeros((self.A.shape[0],1))
            new_column[i] = 1
            if(self.equal[i]==0):#=，引入一个人工变量（大M法，基变量）
                self.A = np.column_stack((self.A,new_column))
                self.object = np.append(self.object,self.M)
                self.Cb[i] = self.M
                self.base_index[i] = int(self.n_params + i)
            elif(self.equal[i]==1):#<=，引入一个松弛变量（基变量）
                self.A = np.column_stack((self.A,new_column))
                self.object = np.append(self.object,0)
                self.base_index[i] = int(self.n_params + i)
            else:# >=,引入一个松弛变量（非基变量），一个人工变量（大M法，基变量）
                new_column2 = np.zeros((self.A.shape[0],1))
                new_column2[i] = -1
                self.A = np.column_stack((self.A,new_column2))
                self.A = np.column_stack((self.A,new_column))
                self.object = np.append(self.object,0)
                self.object = np.append(self.object,self.M)
                self.base_index[i] = int(self.n_params + i + 1)
                self.non_base_index = np.append(self.non_base_index,self.n_params + i)
        self.simplex_table = np.column_stack((self.A,self.b))
    
    def base_choose(self):
        N = self.A[:, self.non_base_index]
        self.check = (self.Cn.T - self.Cb.T @ N)[0] #检验量
        if np.all(self.b>=0):
            index_in = np.argmin(self.check)#,axis=1)
            if self.check[index_in]>=0: #这个时候求解已经完成了
                return -2 #-2是完成，-1是找不到有界解
            index_out = self.cal_argmin(index_in)
            if index_out == -1:
                return -1
        else:#要先用对偶单纯形法满足可行性
            indexs = np.where(self.b<0)[0] #返回的是一个包含着array的元组
            index_out = indexs[0]
            index_in = self.cal_argmin(index_out,is_dual=True)
            if index_in == -3:
                return -3
        return int(self.non_base_index[index_in]),int(self.base_index[index_out]),int(index_in),int(index_out) #返回真实坐标和相对坐标
            
    def solve_problem(self):#todo：行变换可能存在错误
        self.Generate_Simplex_Table()
        while(True):
            in_out = self.base_choose()
            if in_out == -2:#找到解
                print("Problem solved.")
                print(f"The answer:")
                for index,i in zip(self.base_index,range(self.equal.shape[0])):
                    print(f"x{int(index+1)} = {self.b[i]}")
                print(f"The value of the object function:{self.Cb.T @ self.b}")
                break
            elif in_out == -1:#解无界
                print("The problem has no boundary solution.")
                break
            elif in_out == -3:
                print("The problem has no feasible solution.")
                break
            else:
                (index_in,index_out,index_in_nonbase,index_outbase) = in_out
                param_change_cb = self.Cb[index_outbase]
                param_change_cn = self.Cn[index_in_nonbase]
                self.Cn[index_in_nonbase] = param_change_cb #非基变量前系数更新
                self.Cb[index_outbase] = param_change_cn#基变量前系数更新
                column = self.A[:,int(self.non_base_index[index_in])] #注意取的是列
                self.non_base_index[index_in_nonbase] = index_out #序号互换
                self.base_index[index_outbase] = index_in #序号互换
                for i in range(len(column)):
                    if i == index_outbase:
                        self.simplex_table[i] *= (1 / column[i]) #列上的这个元素要变成1
                    else:
                        #print(self.simplex_table[i])
                        self.simplex_table[i] -= column[i] / column[index_outbase] * self.simplex_table[index_outbase] #做初等行变换，列上的其他元素清零，顺便对单纯形表做变换
                self.A,self.b = self.simplex_table[:, :self.A.shape[1]],self.simplex_table[:,self.A.shape[1]:] #变换之后的A，b返回，方便后续判断
                
    def cal_argmin(self,index,is_dual=False):#单纯形法中，是index_in，dual中就是index_out
        min_num = 1e10
        min_index = 0
        if is_dual==False:
            column = self.A[:,int(self.non_base_index[index])]
            if np.all(column<=0)==True:#发现A中对应变量的一列全都小于0，问题的解无界
                return -1
            indexs = np.where(column>0) 
            indexs = indexs[0] #注意indexs是一个元组，第一个元素才是需要的索引数组
            for i in range(len(indexs)):
                if self.b[i] / column[indexs[i]] < min_num:
                    min_num = self.b[i] / column[indexs[i]]
                    min_index = indexs[i]
            return min_index
        else:
            non_base_row = self.A[int(index),self.non_base_index]
            if np.all(non_base_row>=0)==True:
                return -3 #-3表示无可行解
            indexs = np.where(non_base_row<0)
            indexs = indexs[0] #注意indexs是一个元组，第一个元素才是需要的索引数组
            for i in range(len(indexs)):
                if self.check[i] / (-non_base_row[indexs[i]]) < min_num:
                    min_num = self.check[i] / non_base_row[indexs[i]]
                    min_index = indexs[i]
            return min_index
                
A = np.array(([1,2],[2,1],[1,-1]))
b = np.array([4,3,-1])
objectf = np.array([2,1])
equal = np.array([1,1,1])
solver = Simplex(2,objectf,A,b,equal)
solver.solve_problem()
