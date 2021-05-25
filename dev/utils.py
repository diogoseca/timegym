# This file contains utility function to be used during development of timegym.
# This file is not intended for final users.

from IPython.core.magic import register_cell_magic

@register_cell_magic
def runwrite(line, cell):
    """
        This is a magic command for ipython that writes a jupyter cell to a file and runs it at the same time
        In case there's an exception, nothing is written to the file.
        
        Usage:
            %runwrite file.py
            %runwrite -a file.py
            
        Options:
            --append : appends contents of the cell the the file
            -a       : abbreviation for --append
        
        This code was inspired by a snippet written by Andrei Iatsuk on StackOverflow:
        https://stackoverflow.com/questions/33358611/ipython-notebook-writefile-and-execute-cell-at-the-same-time
    """
    argz = line.split()
    file = argz[-1]
    mode = 'w'
    if len(argz) == 2 and ((argz[0] == '-a') or (argz[0] == '--append')):
        mode = 'a'
    try: 
        get_ipython().run_cell(cell)
        with open(file, mode) as f:
            f.write(cell)
    except Exception as ex:
        print(f'Exception found. Nothing written to {file}.')
        raise ex