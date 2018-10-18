from __future__ import print_function
from math import sqrt
from PIL import Image, ImageDraw

def readfile(filename):
    lines=[line for line in open(filename)]

    columnNames=lines[0].strip().split('\t')[1:]
    rowNames=[]
    data=[]
    for line in lines[1:]:
        p=line.strip().split('\t')
        rowNames.append(p[0])
        data.append([float(x) for x in p[1:]])
    return rowNames, columnNames, data

def pearson(v1, v2):
    sum1=sum(v1)
    sum2=sum(v2)

    sum1Sq=sum([pow(v,2) for v in v1])
    sum2Sq=sum([pow(v,2) for v in v2])

    pSum=sum([v1[i]*v2[i] for i in range(len(v1))])

    num=pSum-(sum1*sum2/len(v1))
    den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den==0:
        return 0
    return 1.0-num/den

class biCluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.vec=vec
        self.left=left
        self.right=right
        self.distance=distance
        self.id=id

def hiearachicalCluster(rows, distance=pearson):
    distances={}
    currentClusterId=-1

    clusters=[biCluster(rows[i], id=i) for i in range(len(rows))]

    while len(clusters)>1:
        lowestPair=(0,1)
        closest=distance(clusters[0].vec, clusters[1].vec)

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                if (clusters[i].id, clusters[j].id) not in distances:
                    distances[(clusters[i].id, clusters[j].id)]=distance(clusters[i].vec, clusters[j].vec)
                
                d=distances[(clusters[i].id, clusters[j].id)]
                if d<closest:
                    lowestPair=(i,j)
        
        mergeVector=[(clusters[lowestPair[0]].vec[i]+clusters[lowestPair[1]].vec[i])/2 for i in range(len(clusters[i].vec))]

        newCluster=biCluster(mergeVector, left=clusters[lowestPair[0]], right=clusters[lowestPair[1]], distance=closest, id=currentClusterId)

        currentClusterId-=1
        del clusters[lowestPair[1]]
        del clusters[lowestPair[0]]
        clusters.append(newCluster)
    return clusters[0]

blognames, words, data = readfile('blogdata.txt')
clust=hiearachicalCluster(data)

def printClust(clust, labels=None, n=0):
    for i in range(n):
        print(' ', end='')
    if clust.id<0:
        print('_')
    else:
        if labels==None:
            print(clust.id)
        else:
            print(labels[clust.id])
    
    if clust.left!=None:
        printClust(clust.left, labels=labels, n=n+1)
    if clust.right!=None:
        printClust(clust.right, labels=labels, n=n+1)

def getheight(clust):  
    if clust.left==None and clust.right==None: 
        return 1
    return getheight(clust.left)+getheight(clust.right)

def getdepth(clust):
    if clust.left==None and clust.right==None:
        return 0
    return max(getdepth(clust.left), getdepth(clust.right))+clust.distance

def drawnode(draw,clust,x,y,scaling,labels):  
    if clust.id<0:    
        h1=getheight(clust.left)*20    
        h2=getheight(clust.right)*20 

        top=y-(h1+h2)/2    
        bottom=y+(h1+h2)/2    

        ll=clust.distance*scaling   

        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))

        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)    
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
    else:
        draw.text((x+5,y-7),labels[clust.id],(0,0,0))

def drawdendrogram(clust, labels, jpeg='cluster.jpg'):
    h=getheight(clust)*20
    w=1200
    depth=getdepth(clust)

    scaling=float(w-150)/depth

    img=Image.new('RGB',(w,h),(255,255,255))  
    draw=ImageDraw.Draw(img)

    draw.line((0,h/2,10,h/2),fill=(255,0,0))

    drawnode(draw,clust,10,(h/2),scaling,labels)  
    img.save(jpeg,'JPEG')

# drawdendrogram(clust, blognames, jpeg='blogclust.jpg')

def rotateMatrix(data):
    newdata=[]
    for i in range(len(data[0])):
        newrow=[data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata

rdata=rotateMatrix(data)
wordClust=hiearachicalCluster(rdata)
drawdendrogram(wordClust,labels=words,jpeg='wordclust.jpg')