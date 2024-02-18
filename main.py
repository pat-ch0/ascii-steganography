import argparse
import random as rdm
import re
import subprocess as sub
from PIL import Image,ImageDraw, ImageFont
import numpy as np
import re
import os

def getTuple(l,cherché,index):
    for i in l:
        if i[index] == cherché:
            return i
    return None

def strtoBase(a, base):
    correspondance = "0123456789abcdefghijklmnopqrstuvwxyz"
    ls= []
    for j in a:
        i = ord(j)
        s =""
        while i != 0:
            r = int(i%base)

            s = correspondance[r] + s
            i = (i-r)/base
        ls.append(s)

    return "".join(ls)

def plusProche(str,c,lTriee):
    cL = getTuple(lTriee,c,1)
    newlTri = []
    
    for cara in str:
        if getTuple(newlTri,cara,1) == None:
            newlTri.append(getTuple(lTriee,cara,1))
    
    newlTri.sort(key= lambda x: x[0], reverse=True)
    result = getCharCorrespondPrécis(cL[2],newlTri)

    return result

def getAverageL(image):
  
    im = np.array(image)
    w,h = im.shape    
    return np.average(im.reshape(w*h))

def indicePlusPre(val,lTriee):
    min = (10000,-1)
    for i in range(len(lTriee)):
        if (abs(lTriee[i][0]-val) < min[0]):
            min = (abs(lTriee[i][0]-val),i)

    return min[1]

def redimensionne(im,dim):
    (imX,imY) = im.size
    (x,y) = dim
    lx = []
    ly = []
    diviseurX = imX%x
    diviseurY = imY%y
    ancien = 0
    while ancien < imX:
        ajout = 0
        if diviseurX > 0:
            ajout = 1
        lx.append((ancien,ancien + int(imX/(x)) + ajout))
        ancien += int(imX/(x)) + ajout
        diviseurX -= 1
        
    ancien = 0
    while ancien < imY:
        ajout = 0
        if diviseurY > 0:
            ajout = 1
        ly.append((ancien,ancien + int(imY/(y) + ajout)))
        ancien += int(imY/(y)) + ajout
        diviseurY -= 1

    l = []

    for Y in ly:
        ltemp = []
        for X in lx:
            ltemp.append(int(getAverageL(im.crop((X[0],Y[0],X[1],Y[1])))))
        l.append(ltemp)

    return l


def getCharCorrespondPrécis(img,charTable):
    comp = (1000000,'')
    imgaverageL = getAverageL(img)
    seuil = indicePlusPre(imgaverageL,charTable)
    seuil2 = seuil+4
    if seuil2 >= len(charTable):
        seuil2 = len(charTable)
    seuil1 = seuil-4
    if seuil1 < 0:
        seuil1 = 0

    for x in charTable[seuil1:seuil2]:
        val = 0
        tempx = x[2]
        tempimg = img

        if tempx.size[0] < tempimg.size[0]:
            dimx = tempx.size[0]
        else :
            dimx = tempimg.size[0]

        if tempx.size[1] < tempimg.size[1]:
            dimy = tempx.size[1]
        else:
            dimy = tempimg.size[1]

        tempx = redimensionne(tempx,(dimx,dimy))
        tempimg = redimensionne(tempimg,(dimx,dimy))

        for i in range(len(tempx)):
            for j in range(len(tempx[i])):
                val += abs(tempx[i][j] - tempimg[i][j])
        
        if val < comp[0]:
            comp = (val,x[1])
    
    return comp[1]

def getCharCorrespond(img,charTable):
    comp = (1000000,'')
    imgaverageL = getAverageL(img)
    seuil = int((imgaverageL*(len(charTable)-1))/255)
    seuil2 = seuil+6
    if seuil2 >= len(charTable):
        seuil2 = len(charTable)-1
    seuil1 = seuil-6
    if seuil1 < 0:
        seuil1 = 0

    for x in charTable[seuil1:seuil2]:
        val = 0
        imNp = np.array(img.resize(x[2].size))
        imgNp = np.array(x[2])
        im = imNp - imgNp
        for i in im:
            for j in i:
                val += abs(j)
        if val < comp[0]:
            comp = (val,x[1])

    return comp[1]

def charToImage(font,char,dim,inverse):
    text = ""
    text += char

    font = ImageFont.truetype(font=font,size=dim)
    if inverse:
        img = Image.new('L',font.getsize(text),color="white")
        draw = ImageDraw.Draw(im=img)
        draw.text((0,0), text,(0),font=font)
    else:
        img = Image.new('L',font.getsize(text),color="black")
        draw = ImageDraw.Draw(im=img)
        draw.text((0,0), text,(255),font=font)
    
    return img

def triString(String,imgTable,stringPasUtil):
    liste = []
    liste2 = []
    for i in range(len(String)):
        niveauGris = getAverageL(imgTable[i])
        liste.append((niveauGris,String[i],imgTable[i]))
        boolean = True
        for j in stringPasUtil:
            if j == String[i]:
                boolean=False
        if boolean:
            liste2.append((niveauGris,String[i],imgTable[i]))
    
    liste.sort(key= lambda x: x[0], reverse=False)
    liste2.sort(key= lambda x: x[0], reverse=False)
    return (liste,liste2)

def AfficheIMG(IMG):
    for s in IMG:
        print("".join(s))

def AfficheIMGPlusPos(IMG,pos):
    posAct = 0
    IMGAff = []
    for s in range(len(IMG)):
        tempIMGAff = []
        for c in range(len(IMG[s])):
            if posAct < len(pos) and (c,s) == pos[posAct]:
                tempIMGAff.append(f"\033[42m{IMG[s][c]}\033[0m")
                posAct += 1
            else:
                tempIMGAff.append(IMG[s][c])
        IMGAff.append(tempIMGAff)

    for s in IMGAff:
        print("".join(s))

def contraste(img,c):
    pixels = img.load()
    facteur = (259 * (c+255)) / (255*(259-c))
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            nRed = int(facteur * (pixels[i,j][0] - 128) + 128)
            nGreen = int(facteur * (pixels[i,j][1] - 128) + 128)
            nBlue = int(facteur * (pixels[i,j][2] - 128) + 128)
            pixels[i,j] = (nRed,nGreen,nBlue)

def selectPos(string, c, lTriee,seuil):
    dico={}
    s = ""
    tupleC = getTuple(lTriee,c,1)
    for i in string:
        if getTuple(lTriee,i,1)[0] > tupleC[0]-seuil :
            s += i
        dico[i] = dico.get(i, 0) + 1
    
    if len(s) == 0:
        return None

    choiC = plusProche(s,c,lTriee)

    choiI = rdm.randint(1,dico[choiC])
    for i in range(len(string)):
        if string[i] == choiC:
            if choiI != 1:
                choiI -= 1
            else:
                return i

def répartitionCode(img, bin,lTriee,seuilTolérance):

    def dimto2dim(i,dimX,dimY):
        return(i%dimX,i//dimX)

    dimX = len(img[0])
    dimY = len(img)
    debZone = 0
    pos = 0
    lpos = []
    while pos < len(bin):
        i = len(bin)-pos
        temp = rdm.randint(debZone + int(((dimX*dimY-debZone) / i)/3),int((dimX*dimY-debZone) / i)+debZone)
        s = ""
        if temp > (dimX)*(dimY)-1:
            print("Error impossible to hide the secret sentence please change the seuil or of algorithm")
            exit()

        for j in range(debZone,temp+1):
            (X,Y) = dimto2dim(j,dimX,dimY)
            s += img[Y][X]

        posSelect = selectPos(s,bin[len(bin)-i],lTriee,seuilTolérance)
        
        if posSelect != None:
            (X,Y) = dimto2dim(debZone + posSelect,dimX,dimY)
            lpos.append((X,Y))
            img[Y][X] = bin[len(bin)-i]
            pos = pos +1

        debZone = temp+1   

    return (img,lpos)

def cacheCode2 (img,bin,lTriee,charSép,écart):
    listeCachePotentiel = []
    for y in range(len(img)):
        for x in range(len(img[y])):
            dejaTrouve = False
            for c in charSép:
                if not(dejaTrouve):
                    if x-écart >= 0 and img[y][x-écart] == c and img[y][x] != c:
                        if x+1 > len(img[y])-1 or not(img[y][x+1-écart] == c and img[y][x+1] != c):
                            dejaTrouve = True
                            listeCachePotentiel.append((x,y))
                    elif x+écart < len(img[y]) and img[y][x+écart] == c and img[y][x] != c:
                        if x-1 < 0 or not(img[y][x-1+écart] == c and img[y][x-1] != c):
                            dejaTrouve = True
                            listeCachePotentiel.append((x,y))

    if len(bin) > len(listeCachePotentiel):
        print("Error not enough place to hide the secret sentence. Add character to the algorithm or change the space between the character you want to hide and the reference character.")
        exit()

    debZone = 0
    finZone = 0
    lpos = []
    for b in range(len(bin)):
        debZone = finZone+1
        finZone = debZone + int((len(listeCachePotentiel)-(len(bin)-1))/len(bin))
        (X,Y) = listeCachePotentiel[rdm.randint(debZone,finZone)]
        lpos.append((X,Y))
        img[Y][X] = bin[b]


    return (img,lpos)

descStr = "This program converts an image into ASCII art and can hide a secret sentence in it."
parser = argparse.ArgumentParser(description=descStr)

help = "Path to the image who's gonna be convert to Ascii"
parser.add_argument('-f, --file', dest='imgFile', required=True, help=help)
help = "Number of columns that gonna be used in the Ascii art"
parser.add_argument('-cl, --columns', type = int ,dest="columns", required=False, help=help)
help = "Number of rows that gonna be used in the Ascii art"
parser.add_argument('-r, --rows',type = int,dest="rows",required=False, help=help)
help = "String of all the character that is used in the Ascii art image"
parser.add_argument('-t, --tableAscii',dest="asciiChar",required=False, help=help)
help = "String of name of the output latex file"
parser.add_argument('-o, --output',dest="output",required=False, help=help)
help = "Compile the latex file into a pdf, WARNING you need to use --output to compile the latex file into a pdf"
parser.add_argument('-pdf, --compile',dest="compile",required=False,action='store_true', help=help)
help = "Factor of the image contrast enhancement, the default value is 128,/!\\ the factor need to be between 0 and 258"
parser.add_argument('-co, --contrast',type = int,dest="contrast",required=False, help=help)
help = "Show the image after enhance the contrast"
parser.add_argument('-imsh, --showImage',dest="showImage",required=False,action='store_true', help=help)
help = "Base you want the the secret code need to be encode, /!\\ the base need to be between 2 and 36"
parser.add_argument('-b, --base',dest="base",type = int,required=False,help=help)
help = "Secret sentence you want to hide in the image"
parser.add_argument('-sc, --secretCode',dest="code",required=False,help=help)
help = "Just hide the secret sentence in the image but with all the characters in capital"
parser.add_argument('-U, --upper',dest="upper",required=False,action='store_true',help=help)
help = "Just hide the secret sentence in the image with only lowercase character"
parser.add_argument('-L, --lower',dest="lower",required=False,action='store_true',help=help)
help = "Just hide the secret sentence in the image without any encode"
parser.add_argument('-All, --all',dest="all",required=False,action='store_true',help=help)
help = "Show the position of the characters who need to be found in the image"
parser.add_argument('-pos, --position',dest="position",required=False,action='store_true',help=help)
help = "create the ASCII image to work with white background and black writing"
parser.add_argument('-re, --reverse',dest="reverse",required=False,action='store_true', help=help)
help = "first algorithm to hide your sentence in the Ascii image. You can change the difference between the character you add to write and the original one.The origine value is 20,you can choose any number but under 20 you have less chance to be able to hide your sentence"
parser.add_argument('-Alg1, --algorithme1',dest="algo1seuil",required=False,nargs="*",help=help)
help = "second algorithm to hide your sentence in the Ascii image. You can change the liste of character how is the reference and the distance between the character of the list and the character you want to hide. There is no specific order to write the distance and the list of characters. The original distance is 2 and the characters in the list is just space"
parser.add_argument('-Alg2, --algorithme2',dest="algo2pos",required=False,nargs="*",help=help)

args = parser.parse_args()

if args.algo1seuil and len(args.algo1seuil) > 1:
    parser.error("-Algo1 to many arguments")

if args.algo2pos and len(args.algo2pos) > 2:
    parser.error("-Algo2 to many arguments")

if args.compile and not args.output:
    parser.error("--output required to compile")

File = args.imgFile

if args.contrast and (args.contrast > 258 or args.contrast < 0):
    parser.error("the contrast factor need to be between 0 and 258")

if args.code and not(args.upper or args.lower or args.base or args.all):
    parser.error("You need to say how yout want the secret sentence to be encode")

if not(args.code) and (args.upper or args.lower or args.base or args.all):
    parser.error("You need to enter a secret sentence to choose a way to encode it")

if not(args.code) and args.position:
    parser.error("You need to enter a secret sentence to show the position of the characters who need to be found")

if (args.base and (args.upper or args.lower or args.all)) or (args.upper and (args.base or args.lower or args.all)) or (args.lower and (args.base or args.upper or args.all)) or (args.all and (args.base or args.upper or args.lower)):
    parser.error("You can't use two different encoding to encode the secret sentence")

if args.base and (args.base < 2 or args.base > 36):
    parser.error("The base need to be between 2 and 36")

if args.code and not(args.algo1seuil != None or args.algo2pos != None):
    parser.error("You need to choose which algorithm you want to use to hide your sentence")

asciiChar = "$@%&#*/|(){}[]?-_+~<>!;:,\"^`'. 0123456789AZERTYUIOPQSDFGHJKLMWXCVBNazertyuiopqsdfghjklmwxcvbn"
if args.asciiChar:
    asciiChar = args.asciiChar

nePasUtiliser = ""
if args.code:
    if args.base:
        correspondance = "0123456789abcdefghijklmnopqrstuvwxyz"
        nePasUtiliser = correspondance[0:args.base]
        args.code = strtoBase(args.code,args.base)
    if args.upper:
        nePasUtiliser = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        args.code = args.code.upper()
        args.code = re.sub("\ ","",args.code)
    if args.lower:
        nePasUtiliser = "abcdefghijklmnopqrstuvwxyzç0123456789"
        args.code = args.code.lower()
        args.code = re.sub("\ ","",args.code)
    if args.all:
        nePasUtiliser = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        args.code = re.sub("\ ","",args.code)

font = "Font/Consolas.ttf"
try:
    with open("Configuration.txt", 'r') as f:
        test = f.read().split(" : ")
        if test[0] == "font":
            if os.path.isfile(test[1]):
                font = test[1]
            else:
                print(f"The path to the font is incorrect: {test[1]}")
                exit()
        else:
            print("Error in the configuration file")
            exit()

except FileNotFoundError as e:
    with open("Configuration.txt", 'w') as f:
        f.write("font : "+font)

except IOError as e:
    with open("Configuration.txt", 'w') as f:
        f.write("font : "+font)


fontSize = 14
imageTable = []

for c in (asciiChar+nePasUtiliser):
    imageTable.append(charToImage(font,c,fontSize,args.reverse))

(lTriee,lTrieeAffi) = triString(asciiChar,imageTable,nePasUtiliser)

échelle = 0.5
print()


image = Image.open(File)
contrasteFacteur = 128
if args.contrast:
    contrasteFacteur = args.contrast

contraste(image,contrasteFacteur)

if args.showImage:
    image.show()

image = image.convert('L')

W, H = image.size

columns = os.get_terminal_size()[0]
if args.columns:
    columns = args.columns
    w = W/columns
    h = w/échelle
    rows = int(H/((W/columns)/échelle))
    print(f"{columns}, {W}, {H}, {échelle}, {H/((W/columns)/échelle)}, {H}/(({W}/{columns})/{échelle})")

rows = os.get_terminal_size()[1]
if args.rows:
    rows = args.rows
    columns = int((rows*W)/(H*échelle))
    w = W/columns
    h = w/échelle
    print(f"{rows}, {W}, {H}, {échelle}, {(rows*W)/(H*échelle)}, ({rows}*{W})/({H}*{échelle})")


if columns < int(W/((H/rows)*échelle)):
    w = W/columns
    h = w/échelle
    rows = int(H/((W/columns)/échelle))
else:
    columns = int((rows*W)/(H*échelle))
    w = W/columns
    h = w/échelle

IMG = []

for j in range(rows):
    y1 = int(j*h) 
    y2 = int((j+1)*h)

    if j == rows-1: 
        y2 = H

    out = []

    for i in range(columns):
        x1 = int(i*w) 
        x2 = int((i+1)*w)

        if i == columns-1: 
            x2 = W 

        

        out.append(getCharCorrespond(image.crop((x1,y1,x2,y2)),lTrieeAffi))

    IMG.append(out)

if args.code:
    if len(args.code) > (len(IMG)*len(IMG[0]))/3:
        print("Error the code you try to hide in your image is too long to be hide correctly")
        exit(0)

    if args.algo1seuil != None:
        seuil = 20
        if len(args.algo1seuil) > 0:
            seuil = args.algo1seuil[0]
        (IMG,listePositions) = répartitionCode(IMG,args.code,lTriee,25)
    if args.algo2pos != None:
        if args.reverse:
            if args.all:
                listeCharVisé = "]"
            elif args.upper:
                listeCharVisé = "#"
            else:
                listeCharVisé = "B"
        else:
            listeCharVisé = " "
        distance = 2
        if len(args.algo2pos) == 2:
            if args.algo2pos[0].isnumeric():
                distance = int(args.algo2pos[0])
                listeCharVisé = args.algo2pos[1]
            else:
                distance = int(args.algo2pos[1])
                listeCharVisé = args.algo2pos[0]
        elif len(args.algo2pos) == 1:
            if args.algo2pos[0].isnumeric():
                distance = int(args.algo2pos[0])
            else:
                listeCharVisé = args.algo2pos[0]

        (IMG,listePositions) = cacheCode2(IMG,args.code,lTriee,listeCharVisé,distance)


    AfficheIMG(IMG)
    if args.position:
        AfficheIMGPlusPos(IMG,listePositions)
else:
    AfficheIMG(IMG)



if args.output:
    file = args.output
    exten = ".txt"
    ftext = open(file+exten, "w+", encoding="utf8")
    for i in range(len(IMG)):
        IMG[i] = "".join(IMG[i])
    ftext.write("\n".join(IMG))
    ftext.close()

    if args.position:
        fpos = open(file+"Pos"+exten, "w+", encoding="utf8")
        posAct = 0
        IMGAff = []
        for s in range(len(IMG)):
            tempIMGAff = []
            for c in range(len(IMG[s])):
                if posAct < len(listePositions) and (c,s) == listePositions[posAct]:
                    tempIMGAff.append(f"\\hl{{{IMG[s][c]}}}")
                    posAct += 1
                else:
                    tempIMGAff.append(IMG[s][c])
            IMGAff.append("".join(tempIMGAff))
        fpos.write("\n".join(IMGAff))
        fpos.close()


    strinS = "\\documentclass{article}\n\\usepackage{filecontents,listings,graphicx,varwidth}\n\\usepackage{listings}\n\\usepackage{fullpage}\n\\usepackage{inconsolata}\n\n\n\\lstset{\n  basicstyle=\\ttfamily,\n}"
    strinS += "\n\\newsavebox{\\asciiart}\n\\newcommand{\\Asciifile}{\\raisebox{.8\\height}{\\resizebox{\\textwidth}{!}{\\usebox{\\asciiart}}}}\n\n\\begin{document}\n\\thispagestyle{empty}\n\\begin{lrbox}{\\asciiart}"
    strinS += "\n\\begin{varwidth}{\\maxdimen}\n\\noindent\\lstinputlisting[basicstyle=\\ttfamily]{"
    strinS += file+exten
    strinS +="}\n\\end{varwidth}\n\\end{lrbox}\n\n\\begin{center}\n\\Asciifile{}\n\\end{center}\n\n\\end{document}"

    file = args.output
    file += ".tex"

    ftex = open(file, "w+", encoding="utf8")
    ftex.write(strinS)
    ftex.close()

    if args.compile:
        void = sub.run(["pdflatex",file],capture_output=True)
