# coding=utf-8
import numpy as np


class PerformanceMeasure():
    def __init__(self, real_list=None, pred_list=None, loc=None, percentage=0.2, ranking="defect", cost="module"):
        self.real = real_list
        self.pred = pred_list
        self.loc = loc
        self.percentage = percentage
        self.ranking = ranking
        self.cost=cost

    def getSomePerformance(self):
        if (len(self.pred) != len(self.real)) or (len(self.pred) != len(self.loc) or (len(self.loc) != len(self.real))):
            print("预测缺陷数目或密度与真实缺陷数目或密度，输入长度不一致！")
            exit()

        M = len(self.real)
        L = sum(self.loc)
        P = sum([1 if i > 0 else 0 for i in self.real])
        Q = sum(self.real)


        if (self.ranking == "density" and self.cost=='loc'):
            density = [self.pred[i] / self.loc[i] for i in range(len(self.pred))]
            sort_axis = np.argsort(density)[::-1]
            sorted_pred = np.array(self.pred)[sort_axis]
            sorted_real = np.array(self.real)[sort_axis]
            sorted_loc = np.array(self.loc)[sort_axis]

            locOfPercentage = self.percentage * L
            sum_ = 0
            for i in range(len(sorted_loc)):
                sum_ += sorted_loc[i]
                if (sum_ > locOfPercentage):
                    m = i
                    break
                elif (sum_ == locOfPercentage):
                    m = i + 1
                    break

            PMI=m/M
            PLI=self.percentage


        elif (self.ranking == "defect" and self.cost == 'module'):  # 按照缺陷个数从大到小排序module，检测前percentage的loc

            sort_axis = np.argsort(self.pred)[::-1]
            sorted_pred = np.array(self.pred)[sort_axis]
            sorted_real = np.array(self.real)[sort_axis]
            sorted_loc = np.array(self.loc)[sort_axis]

            m = int(self.percentage * M)
            PMI=self.percentage
            locsum = sum([sorted_loc[i] for i in range(0, m)])
            PLI = locsum / L
        else:
            print("Parameter input error")
            exit()



        tp = sum([1 if sorted_real[j] > 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
        fn = sum([1 if sorted_real[j] > 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
        fp = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
        tn = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
        #print('tp:{0},fn:{1},fp:{2},tn:{3}'.format(tp,fn,fp,tn))

        if (tp+fp==0):
            Precision=0
        else:
            Precision = tp / (tp + fp)

        if(tp+fn==0):
            Recall = 0
        else:
            Recall = tp / (tp + fn)

        if (Recall + Precision == 0):
            F1=0
        else:
            F1 = 2 * Recall * Precision / (Recall + Precision)

        if (tp + fn + fp + tn==0):
            Precisionx=0
        else:
            Precisionx = (tp + fn) / (tp + fn + fp + tn)

        if (P==0):
            Recallx = 0
        else:
            Recallx = (tp + fn) / P

        if (Recallx + Precisionx==0):
            F1x = 0
        else:
            F1x = 2 * Recallx * Precisionx / (Recallx + Precisionx)

        if (fp+tn==0):
            PF = 0
        else:
            PF = fp / (fp + tn)
        if (tp+fp==0):
            falsealarmrate=0
        else:
            falsealarmrate = fp / (tp + fp)

        IFLA = 0
        IFMA = 0
        for i in range(m):
            if (sorted_real[i] > 0):
                break
            else:
                IFLA += sorted_loc[i]

                IFMA += 1


        PofB = sum([sorted_real[j] if sorted_real[j] > 0 else 0 for j in range(m)]) / Q


        if (self.ranking == "density" and self.cost=='loc'):
            PofBPMLI = PofB/PMI
        elif (self.ranking == "defect" and self.cost == 'module'):
            PofBPMLI = PofB / PLI
        else:
            print("Parameter input error")
            exit()

        return Precision, Recall, F1, Precisionx, Recallx, F1x, PF, falsealarmrate, IFMA, IFLA, PMI, PLI, PofB, PofBPMLI





    def PercentPOPT(self):
        M = len(self.real)
        L = sum(self.loc)
        Q = sum(self.real)

        if (self.ranking == "density" and self.cost == 'loc'):

            density = [self.pred[i] / self.loc[i] for i in range(len(self.pred))]

            sort_axis = np.argsort(density)[::-1]
            sorted_loc = np.array(self.loc)[sort_axis]
            locOfPercentage = self.percentage * L
            sum_ = 0
            for i in range(len(sorted_loc)):
                sum_ += sorted_loc[i]
                if (sum_ > locOfPercentage):
                    m = i
                    break
                elif (sum_ == locOfPercentage):
                    m = i + 1
                    break

            realdensity=[self.real[i] / self.loc[i] for i in range(len(self.real))]
            sort_axis = np.argsort(realdensity)[::-1]
            sorted_loc = np.array(self.loc)[sort_axis]
            locOfPercentage = self.percentage * L
            sum_ = 0
            for i in range(len(sorted_loc)):
                sum_ += sorted_loc[i]
                if (sum_ > locOfPercentage):
                    optimalm = i
                    break
                elif (sum_ == locOfPercentage):
                    optimalm = i + 1
                    break

            realdensity = [self.real[i] / self.loc[i] for i in range(len(self.real))]
            sort_axis = np.argsort(realdensity)
            sorted_loc = np.array(self.loc)[sort_axis]
            locOfPercentage = self.percentage * L
            sum_ = 0
            for i in range(len(sorted_loc)):
                sum_ += sorted_loc[i]
                if (sum_ > locOfPercentage):
                    worstm = i
                    break
                elif (sum_ == locOfPercentage):
                    worstm = i + 1
                    break

            pred_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(self.loc, self.pred)]
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            xcost = self.loc
            xcostsum = sum(xcost)
        elif (self.ranking == "defect" and self.cost == 'module'):

            m = int(self.percentage * M)
            optimalm = m
            worstm = m

            pred_index = self.pred
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            xcost = [1 for i in range(len(self.pred))]
            xcostsum = sum(xcost)

        else:
            print("参数传入错误")
            exit()

        optimal_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(xcost, self.real)]
        optimal_index = list(np.argsort(optimal_index))
        optimal_index.reverse()

        optimal_X = [0]
        optimal_Y = [0]
        for i in optimal_index:
            optimal_X.append(xcost[i] / xcostsum + optimal_X[-1])
            optimal_Y.append(self.real[i] / Q + optimal_Y[-1])

        percentoptimal_auc = 0.
        prev_x = 0
        prev_y = 0
        index=0
        for x, y in zip(optimal_X, optimal_Y):
            if x != prev_x:
                percentoptimal_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('percentoptimalx', (x - prev_x))
                #print('percentoptimaly', (y + prev_y))
                prev_x = x
                prev_y = y
                if index==optimalm:
                    break
                index = index + 1

        #print('percentoptimaldauc', percentoptimal_auc)



        pred_X = [0]
        pred_Y = [0]
        for i in pred_index:
            pred_X.append(xcost[i] / xcostsum + pred_X[-1])
            pred_Y.append(self.real[i] / Q + pred_Y[-1])

        percentpred_auc = 0.
        prev_x = 0
        prev_y = 0
        index=0
        for x, y in zip(pred_X, pred_Y):
            if x != prev_x:
                percentpred_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('percentx', (x - prev_x))
                #print('percenty', (y + prev_y))
                prev_x = x
                prev_y = y
                if index == m:
                    break
                index = index + 1
        #print('m=',m,'percentpredauc', percentpred_auc)

        optimal_index.reverse()
        mini_X = [0]
        mini_Y = [0]
        for i in optimal_index:
            mini_X.append(xcost[i] / xcostsum + mini_X[-1])
            mini_Y.append(self.real[i] / Q + mini_Y[-1])

        percentmini_auc = 0.
        prev_x = 0
        prev_y = 0
        index=0
        for x, y in zip(mini_X, mini_Y):
            if x != prev_x:
                percentmini_auc += (x - prev_x) * (y + prev_y) / 2.
                prev_x = x
                prev_y = y
                if index == worstm:
                    break
                index = index + 1

        #print('worstpercent',percentmini_auc)

        percentmini_auc = 1 - (percentoptimal_auc - percentmini_auc)
        percentnormOPT = ((1 - (percentoptimal_auc - percentpred_auc)) - percentmini_auc) / (1 - percentmini_auc)

        return percentnormOPT

    def POPT(self):

        Q = sum(self.real)

        if (self.ranking == "density" and self.cost == 'loc'):
            pred_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(self.loc, self.pred)]
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            xcost=self.loc
            xcostsum=sum(xcost)


        elif (self.ranking == "defect" and self.cost == 'module'):
            pred_index = self.pred
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            xcost = [1 for i in range(len(self.pred))]
            xcostsum = sum(xcost)


        else:
            print("参数传入错误")
            exit()

        optimal_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(xcost, self.real)]
        optimal_index = list(np.argsort(optimal_index))
        optimal_index.reverse()

        optimal_X = [0]
        optimal_Y = [0]
        for i in optimal_index:
            optimal_X.append(xcost[i] / xcostsum + optimal_X[-1])
            optimal_Y.append(self.real[i] / Q + optimal_Y[-1])

        wholeoptimal_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(optimal_X, optimal_Y):
            if x != prev_x:
                wholeoptimal_auc += (x - prev_x) * (y + prev_y) / 2.

                prev_x = x
                prev_y = y


        pred_X = [0]
        pred_Y = [0]
        for i in pred_index:
            pred_X.append(xcost[i]/ xcostsum + pred_X[-1])
            pred_Y.append(self.real[i] / Q + pred_Y[-1])

        wholepred_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(pred_X, pred_Y):
            if x != prev_x:
                wholepred_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('predx', (x - prev_x))
                #print('predy', (y + prev_y))
                prev_x = x
                prev_y = y

        #print('wholepredauc',wholepred_auc)

        optimal_index.reverse()
        mini_X = [0]
        mini_Y = [0]
        for i in optimal_index:
            mini_X.append(xcost[i]/ xcostsum + mini_X[-1])
            mini_Y.append(self.real[i] / Q + mini_Y[-1])

        wholemini_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(mini_X, mini_Y):
            if x != prev_x:
                wholemini_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('wholeworstpredx', (x - prev_x))
                #print('wholeworstpredy', (y + prev_y))
                prev_x = x
                prev_y = y

        #print('worstwholeauc',wholemini_auc)

        wholemini_auc = 1 - (wholeoptimal_auc - wholemini_auc)
        wholenormOPT = ((1 - (wholeoptimal_auc - wholepred_auc)) - wholemini_auc) / (1 - wholemini_auc)

        return wholenormOPT

    def PercentWholeFPA(self):

        M = len(self.real)
        N = sum(self.loc)
        Q = sum(self.real)

        if (self.ranking == "density" and self.cost=='loc'):

            density = [self.pred[i] / self.loc[i] for i in range(len(self.pred))]

            sort_axis = np.argsort(density)[::-1]
            sorted_real = np.array(self.real)[sort_axis]
            sorted_loc = np.array(self.loc)[sort_axis]
            locOfPercentage = self.percentage * N
            sum_ = 0
            for i in range(len(sorted_loc)):
                sum_ += sorted_loc[i]
                if (sum_ > locOfPercentage):
                    m = i
                    break
                elif (sum_ == locOfPercentage):
                    m = i + 1
                    break


        elif (self.ranking == "defect" and self.cost=='loc'):
            sort_axis = np.argsort(self.pred)[::-1]
            sorted_real = np.array(self.real)[sort_axis]
            sorted_loc = np.array(self.loc)[sort_axis]
            locOfPercentage = self.percentage*N
            sum_ = 0
            for i in range(len(sorted_loc)):
                sum_ += sorted_loc[i]
                if (sum_ > locOfPercentage):
                    m = i
                    break
                elif (sum_ == locOfPercentage):
                    m = i + 1
                    break

        elif (self.ranking == "density" and self.cost=='module'):

            density = [self.pred[i] / self.loc[i] for i in range(len(self.pred))]

            sort_axis = np.argsort(density)[::-1]
            sorted_real = np.array(self.real)[sort_axis]
            m = int(self.percentage * M)


        elif (self.ranking == "defect" and self.cost == 'module'):

            sort_axis = np.argsort(self.pred)[::-1]
            sorted_real = np.array(self.real)[sort_axis]
            m = int(self.percentage * M)

        else:
            print("Parameter input error")
            exit()


        PercentFPA = 0
        for i in range(m):
            PercentFPA += ((M - i) * sorted_real[i]) / (M * Q)

        WholeFPA = 0
        for i in range(M):
            WholeFPA += ((M - i) * sorted_real[i]) / (M * Q)


        bestranking = np.array(self.real)[np.argsort(-np.array(self.real))]
        worstranking = np.array(self.real)[np.argsort(np.array(self.real))]


        worstPercentFPA = 0
        for i in range(m):
            worstPercentFPA += ((M - i) * worstranking[i]) / (M * Q)
        bestPercentFPA = 0
        for i in range(m):
            bestPercentFPA += ((M - i) * bestranking[i]) / (M * Q)

        worstWholeFPA = 0
        for i in range(M):
            worstWholeFPA += ((M - i) * worstranking[i]) / (M * Q)

        bestWholeFPA = 0
        for i in range(M):
            bestWholeFPA += ((M - i) * bestranking[i]) / (M * Q)


        normPercentFPA=(PercentFPA-worstPercentFPA)/(bestPercentFPA-worstPercentFPA)
        normWholeFPA = (WholeFPA - worstWholeFPA) / (bestWholeFPA - worstWholeFPA)

        print("sortedreal", sorted_real)
        print("bestranking",bestranking)
        print('worstranking',worstranking)
        print('Percentfpa',PercentFPA)
        print('wholefpa',WholeFPA)
        print('bestPercentfpa',bestPercentFPA)
        print('worstPercentfpa', worstPercentFPA)
        print("bestwholefpa", bestWholeFPA)
        print("worstwholefpa", worstWholeFPA)

        return PercentFPA,normPercentFPA, WholeFPA, normWholeFPA

    def FPA(self):
        '''
        有四个模块m1,m2,m3,m4，真实缺陷个数分别为1，4，2，1,self.real=[1，4，2，1]
        预测出m1缺陷个数为0，m2缺陷个数为3，m3缺陷个数为5，m4缺陷个数为1,self.pred=[0,3,5,1]
        预测出的排序为m3>m2>m4>m1
        fpa=1/4 *1/8 *(4*2+3*4+2*1+1*1)=0.718
        '''
        K = len(self.real)
        N = np.sum(self.real)
        sort_axis = np.argsort(self.pred)
        testBug = np.array(self.real)
        testBug = testBug[sort_axis]
        P = sum(np.sum(testBug[m:]) / N for m in range(K + 1)) / K
        return P









