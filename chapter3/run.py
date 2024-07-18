import os
import train

temperature_list = [0.44, 0.50, 0.56, 0.64] 
temperature = temperature_list[0]

time_index = 7


def main():

    # set where the dataset is located. please change this.
    directory_pattern= "/work/jh230064a/c35000/public_dataset/" #データセットが保存されているディレクトリ

    train_file_pattern = 'T' + '{:.2f}'.format(temperature) + '/train/*tc' +  '{:02d}'.format(time_index) + '*.npz'
    train_file_pattern = os.path.join(directory_pattern, train_file_pattern) #directory_patternと合わせることで，読み込むファイルを指定する
    test_file_pattern = 'T' + '{:.2f}'.format(temperature) + '/test/*tc' + '{:02d}'.format(time_index) + '*.npz'
    test_file_pattern = os.path.join(directory_pattern, test_file_pattern)

    train.train_model(
        p_frac=1.0, 
        temperature=temperature,
        train_file_pattern=train_file_pattern,
        test_file_pattern=test_file_pattern,
        n_epochs=1000,
        max_files_to_load=400,  
        tcl=time_index,
        seed=0)
        
        
if __name__ == '__main__':
    main()
