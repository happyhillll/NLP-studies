import urllib.request
import pandas as pd
import itertools

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

def nsmc_data():
    train_data = pd.read_table('ratings_train.txt')
    train_data['document']=train_data['document'].astype(str)
    train_data['document_split']=train_data['document'].apply(lambda x:x.split())
    # train_data_drop=train_data[['document_split','label']]
    x_train=train_data['document_split'].values
    list_x_train=list(itertools.chain.from_iterable(x_train))
    y_train=train_data['label'].values
    vocab={
        word : i for i, word in enumerate(list_x_train)
    }
               
    return vocab

if __name__ == '__main__':
    print(nsmc_data())

def nsmc_testdata(test_data,test_data_drop):
    test_data = pd.read_table('ratings_test.txt')
    test_data['document']=test_data['document'].astype(str)
    test_data['document_split']=test_data['document'].apply(lambda x:x.split())
    test_data_drop=test_data[['document_split','label']]
    
    return test_data_drop