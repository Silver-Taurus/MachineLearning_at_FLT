''' Python Debugger '''

import pdb

def test_pdb():
    ''' checking the pdb.set_trace() function '''
    pdb.set_trace()
    return "I don't have time!!!"

print(test_pdb())
