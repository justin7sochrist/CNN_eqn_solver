import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import math

def image_slicer():
    tpp = np.zeros((10,28,28,1))
    j=0;i=0;k=0;l=0
    cc=0; cr = False #character count, current character

    img = image.load_img('data/predict1.png',color_mode='grayscale',target_size=(28,112))
    plt.figure()
    plt.imshow(img)

    g = np.zeros(100) #gaps
    cl = np.zeros(80) #counting character length



    img = image.img_to_array(img)
    gc=img.shape[1]*np.ones(10)

    ss = (np.sum(img,0))
    print ("shape of s:",ss.shape)
    print ("ss",ss)
    for i in ss:
        if i<7140 and not cr:
            cr = True
        elif i==7140 and cr:
            cr = False
            gc[cc]=g[int(k/2)]
            print ("over\n",g)
            print ("character:",cc,"space took:",l, "cl",int(cl[0]),int(cl[1]))
            tp = 255*np.ones((28,28,1)) 
            print("size", img[:, int(cl[0]):int(cl[l-1])].shape,"to",tp[:,14-math.floor((l-1)/2):13+math.ceil((l-1)/2)+1].shape)
            tp[:,14-math.floor((l-1)/2):13+math.ceil((l-1)/2)+1] = img[:, int(cl[0]):int(cl[l-1])]
            
            plt.figure()
            plt.imshow(tp)
            plt.show()
            
            print ("cc:",cc,tpp.shape,"tp",tp.shape)
            tpp[cc]=tp
            g = np.zeros(100) #gap reset
            l=0;k=0 #gap and count reset
            cl = np.zeros(20)
            cc+=1
        elif i==7140 and not cr and cc>0:
            g[k]=j
            k+=1
        elif i<7140 and cr:
            print ("l:",l,"j:",j)
            cl[l]=j
            l+=1
        j+=1
    print ("character count",cc)
    print ("centre points:",gc)
    return tpp,cc



def eqnsolve(eq):
    def isy(x): 
        if x == 'y':
            return True 
        else: 
            False
    def isyy(x):
        if x * 1000 % 1000 ==179:
            return True
    
    i=0; y=0; yf = False
 #   eq = np.array(['3','+','2','y','=','10'])
    op = np.array(['0','0','0','0','0','0','0','0','0','0'])
    trm = np.zeros(10)

    #performing multiplication and division
    j=0
    while j < eq.shape[0]:
        a = eq[j]
        if a.isnumeric():
            trm[i] = trm[i]*10 + float(a)
        elif a=='+':
            i+=1
            op[i-1]='+'
        elif a=='-':
            i+=1
            op[i-1]='-'
        elif a=='*':
            if eq[j+1].isnumeric:
                trm[i] = trm[i]*float(eq[j+1])
                j+=1    
        elif  a=='/':
            if eq[j+1].isnumeric:
                trm[i] = trm[i]/float(eq[j+1])
                j+=1 
        elif isy(a):
            print("tt",trm)
            trm[i]+=.179
            print("tta",trm)
            yf = True
        elif a == '=':
            i+=1
            op[i-1]='='
        j+=1
    j=0;y=0; fin =0
    print(trm)
    print(op)

    #performing addition and subs
    trm = trm[:i+1]
    op = op[:i]
    while (j<=i):
        print("trm_inbtw",trm,trm[j])
        if i==j:
            if isyy(trm[j]): y=j
            break
        if isyy(trm[j]):
            y=j
        elif trm[j]=='=':
            pass
        else:
            if op[j]=='+' and not isyy(trm[j+1]):
                tmp = trm[j]+trm[j+1]
                i-=1
                trm = np.delete(trm,j)
                op = np.delete(op,j)
                trm[j]=tmp
            elif op[j] == '-':
                tmp = trm[j]-trm[j+1]
                i-=1
                trm = np.delete(trm,j)
                op = np.delete(op,j)
                trm[j]=tmp
        j+=1

    print(trm.shape[0],"y",y)
    if trm[y] == .179: trm[y]=1.179
    if trm.shape[0]==2:
        if y==0:
            fin =trm[1]/int(trm[0])
        else:
            fin = trm[0]/int(trm[1])
    elif trm.shape[0]==3:
        if y==0:
            print("f1")
            fin = (trm[2]-trm[1])/int(trm[0])
        elif y==2:
            print("f2")
            fin = (trm[0]-trm[1])/int(trm[2])
        elif y==1 and op[1]=='=':
            print("f3")
            fin = (trm[2]-trm[0])/int(trm[1])
        elif y==1 and op[0]=='=':
            print("f4")
            fin = (trm[0]-trm[2])/int(trm[1])
        else:
            fin = trm[0]
        
    print ("final:",fin) 
    print (trm)
    print (op)
    
    
    
    


#main starts
img_height = 28
img_width = 28
batch_size = 2

model = keras.Sequential ([
    layers.Input((28, 28, 1)),
    layers.Conv2D(16,3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(16),
])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'data/mychar/',
    labels='inferred',
    label_mode="int",
    class_names=['0', '1', '2', '3', '4', '5', '6', '7',  '8', '9','add','sub','mul','div','y','eq'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=(True),
    seed=123,
    validation_split=0.1,
    subset="training"
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'data/mychar/',
    labels='inferred',
    label_mode="int",
    class_names=['0', '1', '2', '3', '4', '5', '6', '7',  '8', '9','add','sub','mul','div','y','eq'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=(True),
    seed=123,
    validation_split=0.1,
    subset="validation"
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    metrics=["accuracy"],
)

model.fit(ds_train, epochs = 20, verbose = 2)

model.summary()

#predicting part

img = image.load_img('data/predict.png',color_mode='grayscale',target_size=(img_height, img_width))
plt.figure()
plt.imshow(img)
p=0;c=0
abc,c = image_slicer()
ind = np.array(['0','0','0','0','0','0','0','0','0','0'])
while (p<c):
    ab = abc[p]
    
    # plt.figure()
    # plt.imshow(ab)
    # plt.show()
    
#   stop = input()
    ab = np.expand_dims(ab, axis=0)
    clas =model.predict (ab, batch_size=1)
    s=0;j=0;index=0

    for i in clas[0]:
        print (i)
        if i>s:
            s=i
            index=j
        j+=1
        
    if index==10:
        index = '+'
    elif index == 11:
        index = '-'
    elif index == 12:
        index = '*'
    elif index == 13:
        index = '/'
    elif index == 14:
        index = 'y'
    elif index ==15:
        index = '='
    print ("index:",index)
    print("p:",p)
    ind[p]=index
    p+=1
ind = ind[:7        ]
print(ind)
eqnsolve(ind)



