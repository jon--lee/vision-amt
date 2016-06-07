from PIL import Image

for i in range(1, 10):
    name = "obj" + str(i) + ".png"
    obj = Image.open(name)
    obj = obj.convert('RGBA')
    datas = obj.getdata()
    newData = []
    for item in datas:
        if item[3] != 0:
            newData.append((0, 0, 255, 255))
        else:
            newData.append(item)
    obj.putdata(newData)
    fileout = "objB" + str(i) + ".png"
    obj.save(fileout)
