import os
from PIL import Image, ImageDraw, ImageFont
from random import randint, random, choice, uniform, randrange
import string
import numpy as np
from tqdm import trange
 
def list_files(directory, filetype):
    r = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(filetype):
                r.append(os.path.join(root, name))
    return r

def delete_files(directory):
    for root, dirs, files in os.walk(directory):
        for name in files:
            os.remove(os.path.join(root, name))

charChoice = string.ascii_uppercase + string.digits

delete_files('dwgs')
files = list_files('symbols', '.png')
fonts = list_files('fonts', '.ttf')

symbolEdgeThreshold = 200
symbolEdgeColor = (0, 255, 255) # HSV
symbolEdgeColorRGB = (255, 0, 0)

Ndrawings = 10

for n in trange(Ndrawings):
    xSize = randrange(1000, 2000, 100) # 1600
    ySize = randrange(1000, 2000, 100) # 1200
    borderSize = randrange(300, 501, 100) # 300
    borderWidth = randint(1, 5)
    xGrid = randrange(100, 201, 100) # 100
    yGrid = randrange(100, 201, 100) # 100
    fontSize = randint(10, 15) # 15
    maxRowCount = randint(4, 6) # 4
    lineWidth = randint(1, 5)  
    maxTextLength = randint(5, 7) # 5

    symbolDenseness = uniform(0.2, 0.5) # 0.75
    textDenseness = uniform(0.2, 0.5) # 0.5
    
    xSymbol = xGrid / 2
    ySymbol = yGrid / 2
    
    xSize = xSize + 2 * borderSize
    ySize = ySize + 2 * borderSize
    
    fnt = ImageFont.truetype(fonts[randint(0, len(fonts) - 1)], fontSize)   
    
    img = Image.new('RGB', (xSize, ySize), color = 'white')
    d = ImageDraw.Draw(img)
    d.rectangle([(borderSize / 2, borderSize / 2), 
                 (xSize - borderSize / 2, ySize - borderSize / 2)],
                fill=None,
                outline='black', 
                width=borderWidth)
    
    redPixelList = []
    
    for x in range(borderSize, xSize - borderSize, xGrid):
        for y in range(borderSize, ySize - borderSize, yGrid):
            symbolFlag = False
        
            if random() > (1 - symbolDenseness):       
                symbolFlag = True
                symbolPath = files[randint(0, len(files) - 1)]
                name = symbolPath[symbolPath.find('\\') + 1:
                                  symbolPath.rfind('\\')]
            
                if name != '' and name != 'Drawing':
                    symbol = Image.open(symbolPath).convert('HSV')
                    width, height = symbol.size
                    
                    # Downsize image if too large
                    
                    if width > xSymbol or height > ySymbol:
                        symbol.thumbnail((xSymbol, ySymbol), Image.ANTIALIAS)
                        width, height = symbol.size                         
                    
                    # Find symbol edges & put red pixels over them
                    
                    pixels = symbol.load()
                    
                    for w in range(width):
                        if pixels[w, 0][2] < symbolEdgeThreshold:
                            symbol.putpixel((w, 0), symbolEdgeColor)
                            redPixelList.append((x + w, y, 'N'))
                         
                        if pixels[w, height - 1][2] < symbolEdgeThreshold:
                            symbol.putpixel((w, height - 1), symbolEdgeColor)
                            redPixelList.append((x + w, y + height - 1, 'S'))
                    
                    for h in range(height):
                        if pixels[0, h][2] < symbolEdgeThreshold:
                            symbol.putpixel((0, h), symbolEdgeColor)
                            redPixelList.append((x, y + h, 'W'))
                            
                        if pixels[width - 1, h][2] < symbolEdgeThreshold:
                            symbol.putpixel((width - 1, h), symbolEdgeColor)                    
                            redPixelList.append((x - width - 1, y + h, 'E'))
                            
                    img.paste(symbol, (x, y))
            
            if random() > (1 - textDenseness):                
                if symbolFlag == True:
                    xOffset = width + 5
                else:
                    xOffset = 0
    
                yOffset = 0
                
                for i in range(randint(1, maxRowCount)):
                    text = ''.join(choice(charChoice) for i in 
                                   range(randint(1, maxTextLength)))     
                    
                    d.text((x + xOffset, y + yOffset), text, 
                           font=fnt, fill='black')
                    
                    yOffset += fontSize + 5

    pixels = img.load()
    
    nnList = []
    
    for x, y, D in redPixelList:
        if (x, y) not in nnList:
            minDist = 1e15
            
            xLineEnd = 0
            yLineEnd = 0
            
            for x2, y2, D2 in redPixelList:
                if (x2, y2) not in nnList and \
                   ((D == 'N' and y2 < y) or \
                   (D == 'S' and y2 > y) or \
                   (D == 'W' and x2 < x) or \
                   (D == 'E' and x2 > x)):# and D2 != 'E' 
                    dist = np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
                    
                    if dist < minDist and \
                       (dist > xGrid or dist > yGrid):
                        minDist = dist
                        xLineEnd = x2
                        yLineEnd = y2  
                        
            if minDist < 1e15:
                nnList.append((x, y))
                nnList.append((xLineEnd, yLineEnd))
                
                if x == xLineEnd or y == yLineEnd:
                    d.line([(x, y), (xLineEnd, yLineEnd)],
                            fill='black',
                            width=lineWidth)                    
                else: 
                    if D == 'N' or D == 'S':   
                        d.line([(x, y), (x, yLineEnd)],
                                fill='black',
                                width=lineWidth)
                        
                        d.line([(x, yLineEnd), (xLineEnd, yLineEnd)],
                                fill='black',
                                width=lineWidth)
                    else:
                        d.line([(x, y), (xLineEnd, y)],
                                fill='black',
                                width=lineWidth)
                        
                        d.line([(xLineEnd, y), (xLineEnd, yLineEnd)],
                                fill='black',
                                width=lineWidth)               

    img.save(f'dwgs/{n}.png')

