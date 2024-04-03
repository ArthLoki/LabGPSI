import os


def get_current_path():
    return os.getcwd().replace('\\', '/')


def get_base_path():
    og_path = get_current_path()
    l_path = og_path.split('/')
    base_path = l_path[0]
    for i in range(1, len(l_path)-1):
        base_path += f'/{l_path[i]}'
    return base_path


def get_base_path_v2():
    og_path = get_current_path()
    l_path = og_path.split('/')
    base_path = os.path.join(os.path.join(l_path[0], '/'), '/'.join(l_path[1:-1]))
    return base_path


def get_backspace(og_path, chosen_directory):

    try:
        dirs = og_path.split('/')

        if dirs:
            if chosen_directory in dirs:
                i = dirs.index(chosen_directory)

                if i <= len(dirs)-1:
                    return len(dirs)-i-1
                else:
                    return 0
            else:
                new_path = os.path.join(os.path.join(dirs[0], '/'), '/'.join(dirs[1:-1]))
                # for d in dirs[1:-1]:
                #     new_path += '/{}'.format(d)
                return get_backspace(new_path, chosen_directory)
    except Exception as e:
        print(e)
        return -1


def editPath(ogpath):
    backspace = get_backspace(ogpath, 'ProjetoRedesNeurais')
    if backspace > 0:
        dirs = ogpath.split('/')
        edited_path = f'{dirs[0]}'
        for d in dirs[1:-backspace]:
            edited_path += f'/{d}'
        return edited_path
    return ogpath


def editPathV2(ogpath):
    backspace = get_backspace(ogpath, 'LabGPSI')
    if backspace > 0:
        dirs = ogpath.split('/')
        edited_path = f'{dirs[0]}/{"/".join(dirs[1:-backspace])}'
        return edited_path
    return ogpath


# def get_backspace(og_path, chosen_directory):
#     dirs = og_path.split('/')
#
#     if dirs:
#         if chosen_directory in dirs:
#             i = dirs.index(chosen_directory)
#
#             if i <= len(dirs)-1:
#                 return len(dirs)-i-1
#             else:
#                 return 0
#         else:
#             new_path = dirs[0]
#             for d in dirs[1:-1]:
#                 new_path += '/{}'.format(d)
#             return get_backspace(new_path, chosen_directory)
#     else:
#         return 0
