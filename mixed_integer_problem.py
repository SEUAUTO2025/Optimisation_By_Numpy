from simplex import *
#OK!
def is_integer_number(x):
    return isinstance(x, int) or (isinstance(x, float) and x.is_integer())

class MILP(Simplex):
    def __init__(self,n_params,C,A,b,equal):
        super(MILP,self).__init__(n_params=n_params,C=C,A=A,b=b,equal=equal)
        
    def solve_integer_problem(self):
        while(True):
            self.solve_problem()
            non_integer_index = -1#第一个不是整数的基变量相对下标
            for i in range(len(self.b)):
                if self.base_index[i] < self.n_params and is_integer_number(self.b[i]) == False:#解中有非整数
                    non_integer_indexs = i
                    #既然随便取一个就可以了，这样是否比较优？
            if non_integer_indexs == -1:#全是整数，求解完成
                return self.base_index,self.b,self.Cb.T @ self.b
            self.append_constrain(non_integer_index)
            
    def append_constrain(self,non_integer_index):
        #向下取整，引入松弛变量构造新行
        row_check = np.floor(self.A[non_integer_index].copy()) #深复制
        self.A = np.stack((self.A,row_check),axis=0)#按行拼起来
        self.A = np.stack((self.A,np.zeros((3,),dtype=float)),axis=1)#加一新列，松弛变量引入
        self.b = np.append(self.b,np.floor(self.b[non_integer_index]).item())
        self.base_index = np.append(self.base_index,self.A.size(1)) #更新索引表
        self.simplex_table = np.column_stack((self.A,self.b)) #更新单纯形表
        row_non_integer = self.simplex_table[non_integer_index]#在最后面加了一行，前面的索引是不变的，还可以用之前的索引访问到构造割平面的那一行
        self.simplex_table[-1] -= row_non_integer#初等行变换，重新构造单位阵
        self.A,self.b = self.simplex_table[:, :self.A.shape[1]],self.simplex_table[:,self.A.shape[1]:] #变换之后的A，b返回
        #继续单纯形法迭代，还不满足就继续割平面，直到最后找到解