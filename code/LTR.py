"""
The compound differential evolution (DEA) algorithm optimizes the linear model (Linear Model) for
ranking-oriented optimization (LTR).
"""
# coding=utf-8
import math
import random
import numpy as np
from Origin_PerformanceMeasure import Origin_PerformanceMeasure
from PerformanceMeasure1 import PerformanceMeasure

class LTR:
    def __init__(self,
                 NP=100,
                 F_CR=[(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)],
                 generation=100,
                 len_x=19,
                 value_up_range=10.0,
                 value_down_range=-10.0,
                 X=None,
                 y=None,
                 cost=None,
                 costflag=None,
                 logorlinear=None,
                 metircs="pofb20"):
        """
        Initialize LTR model operations
         :param NP: number of chromosomes
         :param F_CR: Probability of F and CR
         :param generation: number of iterations
         :param len_x: feature length (parameter vector for DEA optimization)
         :param value_up_range: (gene upper limit)
         :param value_down_range: (gene lower limit)
         :param X: feature set (original feature set, including LOC features)
         :param y: label set (label is the number of defects)
         :param cost: cost used for calculation, such as LOC value
         :param costflag: type of cost (LOC code line number type)
         :param logorlinear: optimized model (linear or regression model)
         :param metircs: optimized indicators (pofb20, pofb10, dpa)
        """

        self.NP = NP
        self.F_CR = F_CR
        self.generation = generation
        self.len_x = len_x
        self.value_up_range = value_up_range
        self.value_down_range = value_down_range
        # 总体染色体的初始化操作
        self.np_list = self.initialtion()
        self.training_data_X = X
        self.training_data_y = y
        self.cost = cost
        self.costflag = costflag
        self.logorlinear = logorlinear
        self.metircs = metircs

    def initialtion(self):
        """
        Initialize the gene sequence of the chromosome, randomness
        :return:
        """
        # 染色体序列
        np_list = []
        for i in range(0, self.NP):
            # gene sequence on each chromosome
            x_list = []
            for j in range(0, self.len_x):
                # Each gene takes the value of (down, up)
                x_list.append(self.value_down_range + random.random() *
                              (self.value_up_range - self.value_down_range))
            np_list.append(x_list)

        return np_list

    def substract(self, a_list, b_list):
        return [a - b for (a, b) in zip(a_list, b_list)]

    def add(self, a_list, b_list):

        return [a + b for (a, b) in zip(a_list, b_list)]

    def multiply(self, a, b_list):

        return [a * b for b in b_list]

    def random_distinct_integers(self, number, index=None):
        """
        Randomly generate [number] different numbers, excluding index
         :param number: generated number
         :param index: number not included
         :return:
        """
        res = set()
        # while len(res) != int(number):
        #     res.add(random.randint(0, self.NP - 1))
        while len(res) != int(number):
            if index is not None:
                t = random.randint(0, self.NP - 1)
                if t != index:
                    res.add(t)
            else:
                res.add(random.randint(0, self.NP - 1))
        return list(res)

    def mutation_crossover_one(self, np_list):
        """
       Mutation-Crossover-Way 1
         :param np_list: chromosome sequence
         :return: New chromosome sequence after mutation crossover (chromosome sequence to be selected)
        """
        F_CR = random.choice(self.F_CR)
        F = F_CR[0]
        CR = F_CR[1]

        # 突变序列
        v_list = []
        for i in range(0, self.NP):
            r123 = self.random_distinct_integers(3)
            r1 = r123[0]
            r2 = r123[1]
            r3 = r123[2]
            # mutations on each chromosome
            sub = self.substract(np_list[r2], np_list[r3])
            mul = self.multiply(F, sub)
            add = self.add(np_list[r1], mul)
            # 保证染色体突变后，里面的基因还在上下限中
            for i in range(self.len_x):
                if add[i] > self.value_up_range or add[i] < self.value_down_range:
                    add[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)

            v_list.append(add)

        # crossover
        # Cross the mutant sequence with the original series to obtain the chromosome sequence to be selected
        u_list = self.crossover(np_list, v_list, CR)
        return u_list

    def mutation_crossover_two(self, np_list):

        F_CR = random.choice(self.F_CR)
        F = F_CR[0]
        CR = F_CR[1]
        F1 = random.random()

        v_list = []
        for i in range(0, self.NP):
            r12345 = self.random_distinct_integers(5)
            r1 = r12345[0]
            r2 = r12345[1]
            r3 = r12345[2]
            r4 = r12345[3]
            r5 = r12345[4]

            sub1 = self.substract(np_list[r2], np_list[r3])
            sub2 = self.substract(np_list[r4], np_list[r5])
            mul1 = self.multiply(F1, sub1)
            mul2 = self.multiply(F, sub2)
            add1 = self.add(np_list[r1], mul1)
            add2 = self.add(add1, mul2)

            for i in range(self.len_x):
                if add2[i] > self.value_up_range or add2[i] < self.value_down_range:
                    add2[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)
            v_list.append(add2)

        u_list = self.crossover(np_list, v_list, CR)
        return u_list

    def mutation_crossover_three(self, np_list):

        F_CR = random.choice(self.F_CR)
        F = F_CR[0]

        v_list = []
        for i in range(0, self.NP):
            r123 = self.random_distinct_integers(3)
            r1 = r123[0]
            r2 = r123[1]
            r3 = r123[2]
            sub1 = self.substract(np_list[r2], np_list[r3])
            sub2 = self.substract(np_list[r1], np_list[i])
            mul1 = self.multiply(F, sub1)
            mul2 = self.multiply(random.random(), sub2)
            add1 = self.add(mul1, mul2)
            add2 = self.add(add1, np_list[i])

            for i in range(self.len_x):
                if add2[i] > self.value_up_range or add2[i] < self.value_down_range:
                    add2[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)
            v_list.append(add2)

        return v_list

    def crossover(self, np_list, v_list, CR):
        """
        Crossover operation between mutated sequence and original sequence
         :param np_list: original sequence
         :param v_list: mutation sequence
         :param CR: crossover probability
         :return: sequence to be selected
        """
        u_list = []
        for i in range(0, self.NP):
            # 每个染色体交叉后的基因序列
            vv_list = []
            jrand = random.randint(0, self.len_x - 1)
            for j in range(0, self.len_x):
                if (random.random() <= CR) or (j == jrand):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])

            # tmp = random.randint(0, self.len_x - 1)
            # vv_list[tmp] = v_list[i][tmp]
            u_list.append(vv_list)
        return u_list

    def selection(self, u_list1, u_list2, u_list3, np_list):
        """
        In compound differential evolution, the best chromosome is selected from the original sequence and multiple sequences to be selected.
        :param u_list1:
        :param u_list2:
        :param u_list3:
        :param np_list:
        :return:
        """
        # Pick the best chromosome for each
        for i in range(0, self.NP):
            fpa1 = self.Objfunction(u_list1[i])
            fpa2 = self.Objfunction(u_list2[i])
            fpa3 = self.Objfunction(u_list3[i])
            fpa4 = self.Objfunction(np_list[i])
            max_fpa = max(fpa1, fpa2, fpa3, fpa4)
            if max_fpa == fpa1:
                np_list[i] = u_list1[i]
            elif max_fpa == fpa2:
                np_list[i] = u_list2[i]
            elif max_fpa == fpa3:
                np_list[i] = u_list3[i]
            else:
                np_list[i] = np_list[i]
        return np_list

    def process(self):

        np_list = self.np_list
        # The best chromosome (gene sequence)
        max_x = []
        # Get the best indicator result
        max_f = []
        # The value of objFunction for each chromosome
        xx = []
        for i in range(0, self.NP):
            xx.append(self.Objfunction(np_list[i]))

        max_f.append(max(xx))
        max_x.append(np_list[xx.index(max(xx))])

        for i in range(0, self.generation):
            u_list1 = self.mutation_crossover_one(np_list)
            u_list2 = self.mutation_crossover_two(np_list)
            u_list3 = self.mutation_crossover_three(np_list)

            np_list = self.selection(u_list1, u_list2, u_list3, np_list)
            xx = []
            for i in range(0, self.NP):
                xx.append(self.Objfunction(np_list[i]))
            max_f.append(max(xx))
            max_x.append(np_list[xx.index(max(xx))])

        max_ff = max(max_f)

        max_xx = max_x[max_f.index(max_ff)]
        print('the maximum point x =', max_xx)
        print('the maximum value y =', max_ff)

        return max_xx

    def Objfunction(self, Param):
        pred_y = []
        if (self.logorlinear == 'linear'):
            for test_x in self.training_data_X:
                pred_y.append(float(np.dot(test_x, Param)))

        elif (self.logorlinear == 'log'):
            for test_x in self.training_data_X:
                try:
                    res = 1 / (1 + math.exp(-np.dot(test_x, Param)))
                except Exception as e:
                    res = 0
                pred_y.append(res)
        else:
            exit()

        if (self.costflag == 'loc'):
            pred_y = pred_y * np.array(self.cost)
            real_y = self.training_data_y * np.array(self.cost)

            if self.metircs == "pofb20":
                performancevalue = \
                    Origin_PerformanceMeasure(real_y, pred_y, self.cost, self.cost, 0.2, 'density',
                                              'loc').getSomePerformance()[
                        14]
            elif self.metircs == "pofb10":
                performancevalue = \
                    Origin_PerformanceMeasure(real_y, pred_y, self.cost, self.cost, 0.1, 'density',
                                              'loc').getSomePerformance()[
                        14]
            elif self.metircs == "dpa":
                performancevalue = \
                    Origin_PerformanceMeasure(real_y, pred_y, self.cost, self.cost, 0.2, 'density',
                                              'loc').DPA()
        elif (self.costflag == 'cc'):
            pred_y = pred_y * np.array(self.cost)
            real_y = self.training_data_y * np.array(self.cost)
            performancevalue = PerformanceMeasure(real_y, pred_y, self.cost, self.cost, 1, 'complexitydensity',
                                                  'cc').POPT()
        elif (self.costflag == 'module'):
            performancevalue = PerformanceMeasure(self.training_data_y, pred_y).FPA()
        else:

            exit()
        return performancevalue

    def predict(self, testing_data_X, Param):
        pred_y = []
        for test_x in testing_data_X:
            pred_y.append(float(np.dot(test_x, Param)))

        return pred_y

