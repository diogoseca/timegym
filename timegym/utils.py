# this code snippet written by Andrei Iatsuk on StackOverflow:
# https://stackoverflow.com/questions/33358611/ipython-notebook-writefile-and-execute-cell-at-the-same-time
# this writes a jupyter cell to a file and runs it at the same time. 
# this way, we can build and test the package as we develop

from IPython.core.magic import register_cell_magic

@register_cell_magic
def runwrite(line, cell):
    argz = line.split()
    file = argz[-1]
    mode = 'w'
    if len(argz) == 2 and argz[0] == '-a':
        mode = 'a'
    try: 
        #exec(cell)
        get_ipython().run_cell(cell)
        with open(file, mode) as f:
            f.write(cell)
    except Exception as ex:
        raise ex
        print('Nothing written to file.')