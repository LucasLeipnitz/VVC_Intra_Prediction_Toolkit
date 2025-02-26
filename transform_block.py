import numpy as np
import pandas as pd
import math as mh
import re
from collections import defaultdict

path = "./"
path_equations = "output/equations/"

class TransformBlock:
    '''Attributes:
        int nTbW
        int nTbH
        int predModeIntra
        int intraPredAngle
        int refidx
        int refW
        int refH
        int cidx
        int refFilterFlag
        int{ref_index} ref
        int[] ref_id
        int[] array_ifact
        int[] array_iidx
    '''

    
    def __init__(self, nTbW, nTbH, predModeIntra, intraPredAngle, refidx = 0, refW = 0, refH = 0, cidx = 0):
        #Initialize inputs
        self.nTbW = nTbW
        self.nTbH = nTbH
        self.predModeIntra = predModeIntra
        self.intraPredAngle = intraPredAngle
        self.refidx = refidx
        self.refW = refW
        self.refH = refH
        self.cidx = cidx
        

        #refFilterFlag hardcoded for now
        self.refFilterFlag = 1

        #Initialize reference as an empty list
        self.ref = defaultdict(lambda: "Null")
        self.ref_id = []

        #Initialize ifact and iidx array as empty lists
        self.array_ifact = []
        self.array_iidx = []

        self.equations = []
        self.equations_reuse = []

    def calculate_reference_sample_array_greather_equal_34(self):
        index_x = 0
        index_y = - 1 - self.refidx

        #with x = 0...nTbW + refidx + 1. The +1 on the end is to include (nTbW + refidx + 1) in the array
        for x in range((self.nTbW + self.refidx + 1) + 1):
            #ref[x] = p[-1 -refidx + x][-1 -refidx] 
            index_x = -1 - self.refidx + x
            self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
            self.ref_id.append(x)

        if (self.intraPredAngle < 0):
            invAngle = round((512*32)/self.intraPredAngle)                
            index_x = - 1 - self.refidx
            index_y = 0

            #with x = -nTbH ... -1
            for x in range(-self.nTbH, 0):
                #ref[x] = p[-1 -refidx][-1 -refidx + Min((x*invAngle + 256) >> 9, nTbH)]
                index_y = -1 - self.refidx + min((x*invAngle + 256) >> 9, self.nTbH)
                self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)
                    
        else:
            index_y = - 1 - self.refidx
            #with x = nTbW + 2 + refidx ... refW + refidx
            for x in range(self.nTbW + 2 + self.refidx, (self.refW + self.refidx) + 1):
                #ref[x] = p[-1 -refidx + x][-1 -refidx]
                index_x = -1 - self.refidx + x
                self.ref[x] = ("p[" + str(index_x) + "][" + str(index_y) + "]")
                self.ref_id.append(x)
            
            index_x = -1 + self.refW
            #with x = 1...(Max(1,nTbW/nTbH)*refidx + 1)
            for x in range(1,(max(1,self.nTbW/self.nTbH)*self.refidx + 1) + 1):
                #ref[refW + refidx + x] = p[-1 -refW][-1 -refidx]
                self.ref[self.refW + self.refidx + x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)

        self.ref_id.sort()

    def calculate_reference_sample_array_less_34(self):
        index_x = - 1 - self.refidx
        index_y = 0

        #with x = 0...nTbH + refidx + 1. The +1 on the end is to include (nTbH + refidx + 1) in the array
        for x in range((self.nTbH + self.refidx + 1) + 1):
            #ref[x] = p[-1 -refidx][-1 -refidx + x] 
            index_y = -1 - self.refidx + x
            self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
            self.ref_id.append(x)

        if (self.intraPredAngle < 0):
            invAngle = round((512*32)/self.intraPredAngle)                
            index_x = 0
            index_y = - 1 - self.refidx

            #with x = -nTbW ... -1
            for x in range(-self.nTbW, 0):
                #ref[x] = p[-1 -refidx + Min((x*invAngle + 256) >> 9, nTbW][-1 -refidx]
                index_x = -1 - self.refidx + min((x*invAngle + 256) >> 9, self.nTbH)
                self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)
                    
        else:
            index_x = - 1 - self.refidx
            #with x = nTbH + 2 + refidx ... refH + refidx
            for x in range(self.nTbH + 2 + self.refidx, (self.refH + self.refidx) + 1):
                #ref[x] = p[-1 -refidx][-1 -refidx + x]
                index_y = -1 - self.refidx + x
                self.ref[x] = ("p[" + str(index_x) + "][" + str(index_y) + "]")
                self.ref_id.append(x)
            
            index_y = -1 + self.refH
            #with x = 1...(Max(1,nTbH/nTbW)*refidx + 1)
            for x in range(1,(max(1,self.nTbH/self.nTbW)*self.refidx + 1) + 1):
                #ref[refH + refidx + x] = p[-1 -refidx][-1 -refH]
                self.ref[self.refH + self.refidx + x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)

        self.ref_id.sort()


    def calculate_pred_values(self):
        if (self.predModeIntra >= 34):
            self.calculate_reference_sample_array_greather_equal_34()
        else:
            self.calculate_reference_sample_array_less_34()
            
    def calculate_constants_mode(self):
        for x in range(self.nTbW):
            iidx = ((x + 1)*self.intraPredAngle) >> 5
            ifact = ((x + 1)*self.intraPredAngle) & 31
            #print("When x = " + str(x) + ", f = ((" + str(x) + " + 1)* " + str(angle) + ") & 31"," = ",ifact)
            self.array_iidx.append(iidx)
            self.array_ifact.append(ifact)

    def calculate_equations_mode(self):
        columns = []
        for x in range(self.nTbW): 
            columns.append(x)
            current_column = []
            iidx = ((x + 1)*self.intraPredAngle) >> 5
            ifact = ((x + 1)*self.intraPredAngle) & 31
            for y in range(self.nTbH):
                current_column.append("fC[" + str(ifact) + "][0]*ref[" + str(y + iidx + 0) + "] + " + "fC[" + str(ifact) +
                                        "][1]*ref[" + str(y + iidx + 1) + "] + " + "fC[" + str(ifact) + "][2]*ref[" +
                                        str(y + iidx + 2) + "] + " + "fC[" + str(ifact) + "][3]*ref[" + str(y + iidx + 3) + "]")        
            self.equations.append(current_column)

        df = pd.DataFrame(list(zip(*self.equations)),columns = columns)
        excel_writer = pd.ExcelWriter(path + path_equations + "equations_" + str(self.predModeIntra) + "_" + str(self.nTbW) + "x" + str(self.nTbH) + ".xlsx", engine='xlsxwriter') 
        df.to_excel(excel_writer, sheet_name='equations', index=False, na_rep='NaN')

        # Auto-adjust columns' width
        for column in df:
            column_width = 70
            col_iidx = df.columns.get_loc(column)
            excel_writer.sheets['equations'].set_column(col_iidx, col_iidx, column_width)

        excel_writer._save()
        return self.equations

    '''def calculate_equation_with_reuse(self, buffer, x):
        columns.append(x)
        current_column = []
        iidx = ((x + 1)*self.intraPredAngle) >> 5
        ifact = ((x + 1)*self.intraPredAngle) & 31
        for y in range(self.nTbH):
            if(ifact in buffer):
                if((y + iidx) in buffer[ifact]):
                    current_column.append("reuso: " + str(buffer[ifact][y + iidx]))   
                else:
                    current_column.append("fC[" + str(ifact) + "][0]*ref[" + str(y + iidx + 0) + "] + " + "fC[" + str(ifact) +
                                            "][1]*ref[" + str(y + iidx + 1) + "] + " + "fC[" + str(ifact) + "][2]*ref[" +
                                            str(y + iidx + 2) + "] + " + "fC[" + str(ifact) + "][3]*ref[" + str(y + iidx + 3) + "]")
                    buffer[ifact][y + iidx] = str(self.predModeIntra) + " : " + str(x)
            else:
                current_column.append("fC[" + str(ifact) + "][0]*ref[" + str(y + iidx + 0) + "] + " + "fC[" + str(ifact) +
                                            "][1]*ref[" + str(y + iidx + 1) + "] + " + "fC[" + str(ifact) + "][2]*ref[" +
                                            str(y + iidx + 2) + "] + " + "fC[" + str(ifact) + "][3]*ref[" + str(y + iidx + 3) + "]")
                buffer[ifact][y + iidx] = str(self.predModeIntra) + " : " + str(x) + "," + str(y)

            self.equations_reuse.append(current_column)'''
    
    
    def print_reference_sample_array(self):
        ref = []
        for i in self.ref:
            ref.append((i, self.ref[i]))

        sorted_ref = sorted(ref)
        for i in sorted_ref:
            print(i)

    def transform_dict_to_array(self, begin, end, normalize):
        ref = []
        for i in range(begin, end + 1):
            last_iidx = (((self.nTbW - 1) + 1)*self.intraPredAngle) >> 5 #iidx da última posição de x ou y
            if(last_iidx < 0):
                if(not(normalize)):
                    if(i < last_iidx):
                        ref.append("NU")
                    else:
                        ref.append(self.ref[i])
                else:
                    ref.append(self.ref[i])
            else:
                if(not(normalize)):
                    if(i > (last_iidx + (self.nTbW - 1) + 3)):
                        ref.append("NU")
                    else:
                        ref.append(self.ref[i])
                else:
                    ref.append(self.ref[i])
        return ref
            

    def normalize_ref(self):
        for i in self.ref.keys():
            x,y = re.findall(r'\d+', self.ref[i]) #Get x and y value from string

            if(self.predModeIntra >= 34):
                index = y
            else:
                index = x

            if(i < 0):
                if(index != str(abs(i) - 1)):
                    self.ref[int('-' + str(int(index) + 1))] = self.ref[i]
                    self.ref[i] = 'NU'