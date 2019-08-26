import os
import shutil
class CasesSuit:


    def add_cases(self, case_dir):
        ''' 加载 ../test_cases 文件夹下测试用例到 ../data/tmp '''
        # tmp_save_case_folder = '/media/wan/36B0ED8054668DCA/dataset/train/happy'
        # if not os.path.exists(tmp_save_case_folder):
        #     os.mkdir(tmp_save_case_folder)
        list_dir = os.listdir(case_dir)
        for i in list_dir:
            case_file = os.path.join(case_dir, i)
            if os.path.isfile(case_file):
                if 'afe_an' in os.path.split(case_file)[1] :
                    print(case_file)
                #     shutil.copy(case_file,tmp_save_case_folder)
                    os.remove(case_file)
            # if os.path.isdir(case_file):
            #     self.add_cases(case_file)

    # def del_cases(self, tmp_case_dir):
    #     ''' 删除 ../data/tmp 临时文件夹下所有测试用例 '''
    #     filedir = os.listdir(tmp_case_dir)
    #     for filename in filedir:
    #         file = os.path.join(tmp_case_dir, filename)
    #         if os.path.isdir(file):
    #             self.del_cases(file)
    #         else:
    #             os.remove(file)


sol = CasesSuit()
sol.add_cases('/media/wan/36B0ED8054668DCA/dataset/train/happy')