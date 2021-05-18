import numpy as np

'''
Printing functions
'''

def ibordered_title(text):
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ['**' + '*' * width + '**']
    for s in lines:
        res.append('* ' + (s + ' ' * width)[:width] + ' *')
        #res.append(7 * ' ' +(s + ' ' * (width))[:width])
    res.append('**' + '*' * width + '**')
    return '\n'.join(res)

def ibordered(text):
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ['+-' + '-' * width + '-+']
    for s in lines:
        res.append('| ' + (s + ' ' * width)[:width] + ' |')
    res.append('+-' + '-' * width + '-+')
    return '\n'.join(res)

def iprint_line():
    iline = "------------------------------------------------------------"
    return iline
