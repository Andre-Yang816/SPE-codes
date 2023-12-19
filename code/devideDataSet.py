import os
import shutil


def crossRelease(path,write_path):
    #print(path)
    for root, dirs, files in os.walk(path):
        #print(root)
        #print(files)
        files_count = len(files)
        for i in range(files_count-1):
            if files[i][:3] == files[i+1][:3]:
                file1 = os.path.join(root,files[i])
                file2 = os.path.join(root, files[i+1])
                temp_path = files[i][:-4]+'_'+files[i+1][:-4]
                new_write_path = os.path.join(write_path,temp_path)
                #print(new_write_path)

                if not os.path.exists(new_write_path):
                    os.makedirs(new_write_path)

                shutil.copy(file1, new_write_path)
                shutil.copy(file2, new_write_path)


def crossProject(path):
    print(path)


if __name__ == '__main__':
    release_path = '../crossrelease_csv'
    project_path = '../crossproject_csv'
    data_path = '../DataSet'

    crossRelease(data_path, release_path)