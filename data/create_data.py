'''
format
x_1 x_2 x_3 x_4 y for WT and MUTANT 
'''
import random
import math

def main(mutant_file, wt_file,proportion):
    data1 = []
    data2 = []
    data3= []
    with open(mutant_file,'r') as mf:
        for line in mf:
            cnst = line.strip().split(',')
            kcat = cnst[0]
            km = cnst[1]
            err_kcat = cnst[2]
            err_km = cnst[3]
            data1.append('{},{},0.0\n'.format(kcat,km))
            data2.append('{},{},0.0\n'.format(err_kcat,err_km))
            data3.append('{},{},{},0.0\n'.format(kcat,km,math.sqrt(float(err_kcat)**2 + float(err_km)**2)))

    
    with open(wt_file,'r') as wtf:
        for line in wtf:
            cnst = line.strip().split(',')
            kcat = cnst[0]
            km = cnst[1]
            err_kcat = cnst[2]
            err_km = cnst[3]
            data1.append('{},{},1.0\n'.format(kcat,km))
            data2.append('{},{},1.0\n'.format(err_kcat,err_km))
            data3.append('{},{},{},1.0\n'.format(kcat,km,math.sqrt(float(err_kcat)**2 + float(err_km)**2)))
    
    random.shuffle(data1)
    random.shuffle(data2)
    random.shuffle(data3)

    with open('data_kckm_train.csv','w') as train:
        train.write('x_1,x_2,y\n')
        for i in range(int(len(data1)*proportion)):
            train.write(str(data1[i]))
    
    with open('data_kckm_valid.csv','w') as train:
        train.write('x_1,x_2,y\n')
        for i in range(int(len(data1)*proportion)+1, len(data1)):
            train.write(str(data1[i]))
    
    with open('data_err_train.csv','w') as train:
        train.write('x_1,x_2,y\n')
        for i in range(int(len(data2)*proportion)):
            train.write(str(data2[i]))
    
    with open('data_err_valid.csv','w') as train:
        train.write('x_1,x_2,y\n')
        for i in range(int(len(data2)*proportion)+1, len(data2)):
            train.write(str(data2[i]))
    
    with open('data_kckm_err_train.csv','w') as train:
        train.write('x_1,x_2,x_3,y\n')
        for i in range(int(len(data3)*proportion)):
            train.write(str(data3[i]))
    
    with open('data_kckm_err_valid.csv','w') as train:
        train.write('x_1,x_2,x_3,y\n')
        for i in range(int(len(data3)*proportion)+1, len(data3)):
            train.write(str(data3[i]))
    

if __name__ == '__main__':
    main('MU_X.csv', 'WT_X.csv',0.75)

    
