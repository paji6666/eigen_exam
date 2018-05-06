
def substr(strlist,key,subdic):
    returnlist = []
    for item in subdic:
        for s in strlist:
            returnlist.append(s.replace(key,item,1))
    return returnlist
def substr2(strlist,key,subdic):
    while True:
        templist = set(substr(strlist,key,subdic))
        if templist == strlist:
            break
        else:
            strlist = templist
    return strlist
def main(str,dic):
    returnlist = [str]
    for key in dic:
        returnlist = substr2(returnlist,key,dic[key])
    return returnlist



if __name__ == '__main__':
    str = 'adcbf'
    dic =  {'a': ['B', 'C'], 'b': ['X']}

    print(main(str,dic))


